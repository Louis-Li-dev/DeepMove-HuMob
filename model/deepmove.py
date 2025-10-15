from __future__ import print_function
from __future__ import division
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
import pandas as pd
from datetime import datetime

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
class Attn(nn.Module):
    """Batch-compatible Attention Module"""

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history, history_mask=None):
        """
        out_state: (batch, target_len, hidden)
        history: (batch, hist_len, hidden)
        history_mask: (batch, hist_len) - 1 for valid, 0 for padding
        Returns: (batch, target_len, hist_len)
        """
        batch_size = out_state.size(0)
        target_len = out_state.size(1)
        hist_len = history.size(1)
        
        if self.method == 'dot':
            # (batch, target_len, hidden) @ (batch, hidden, hist_len)
            attn_energies = torch.bmm(out_state, history.transpose(1, 2))
        elif self.method == 'general':
            # Transform history: (batch, hist_len, hidden)
            energy = self.attn(history)
            # (batch, target_len, hidden) @ (batch, hidden, hist_len)
            attn_energies = torch.bmm(out_state, energy.transpose(1, 2))
        elif self.method == 'concat':
            # Expand for broadcasting
            out_state_exp = out_state.unsqueeze(2).expand(-1, -1, hist_len, -1)
            history_exp = history.unsqueeze(1).expand(-1, target_len, -1, -1)
            # Concat and transform
            combined = torch.cat([out_state_exp, history_exp], dim=-1)
            energy = self.attn(combined)  # (batch, target_len, hist_len, hidden)
            attn_energies = torch.matmul(energy, self.other)  # (batch, target_len, hist_len)
        
        # Apply mask before softmax
        if history_mask is not None:
            mask = history_mask.unsqueeze(1).expand(-1, target_len, -1)
            attn_energies = attn_energies.masked_fill(mask == 0, -1e9)
        
        return F.softmax(attn_energies, dim=2)


class TrajPreLocalAttnLong(nn.Module):
    """Batch-compatible RNN model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreLocalAttnLong, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.user_size = parameters.user_size
        self.user_emb_size = parameters.user_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type
        self.sos_token = self.loc_size  # SOS token index

        self.emb_loc = nn.Embedding(self.loc_size + 1, self.loc_emb_size, padding_idx=0)  # +1 for SOS
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size, padding_idx=0)
        self.emb_user = nn.Embedding(self.user_size, self.user_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size + self.user_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1, batch_first=True)
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(input_size, self.hidden_size, 1, batch_first=True)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1, batch_first=True)
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1, batch_first=True)
            self.rnn_decoder = nn.RNN(input_size, self.hidden_size, 1, batch_first=True)

        self.fc_final = nn.Linear(2 * self.hidden_size, self.loc_size)  # Output to loc_size (excluding SOS)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, tim, uid, target_len, seq_len):
        """
        Batched forward pass
        loc: (batch, max_seq_len)
        tim: (batch, max_seq_len)
        uid: (batch,)
        target_len: (batch,) - length of target sequence for each sample
        seq_len: (batch,) - total sequence length for each sample (history + target)
        """
        batch_size = loc.size(0)
        max_seq_len = loc.size(1)
        
        # Embeddings
        loc_emb = self.emb_loc(loc)  # (batch, seq, loc_emb)
        tim_emb = self.emb_tim(tim)  # (batch, seq, tim_emb)
        user_emb = self.emb_user(uid).unsqueeze(1).expand(-1, max_seq_len, -1)  # (batch, seq, user_emb)
        
        x = torch.cat([loc_emb, tim_emb, user_emb], dim=2)
        x = self.dropout(x)
        
        # Split into history and current
        # For each sample, history = x[:, :seq_len-target_len], current = x[:, -target_len:]
        # This is tricky with variable lengths, so we'll use masking
        
        # Create masks
        max_target_len = target_len.max().item()
        hist_lens = seq_len - target_len
        max_hist_len = hist_lens.max().item()
        
        # Extract history and current sequences
        history_seqs = []
        current_seqs = []
        hist_mask = torch.zeros(batch_size, max_hist_len, dtype=torch.bool, device=loc.device)
        curr_mask = torch.zeros(batch_size, max_target_len, dtype=torch.bool, device=loc.device)
        
        for i in range(batch_size):
            hl = hist_lens[i].item()
            tl = target_len[i].item()
            
            # History
            if hl > 0:
                history_seqs.append(x[i, :hl])
                hist_mask[i, :hl] = 1
            else:
                history_seqs.append(torch.zeros(1, x.size(2), device=x.device))
            
            # Current (last target_len items)
            current_seqs.append(x[i, hl:hl+tl])
            curr_mask[i, :tl] = 1
        
        # Pad sequences
        history = torch.nn.utils.rnn.pad_sequence(history_seqs, batch_first=True)
        current = torch.nn.utils.rnn.pad_sequence(current_seqs, batch_first=True)
        
        # Pack sequences for RNN
        hist_lens_clamped = hist_lens.clamp(min=1).cpu()
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(
            history, hist_lens_clamped, batch_first=True, enforce_sorted=False
        )
        
        curr_lens_clamped = target_len.clamp(min=1).cpu()
        current_packed = torch.nn.utils.rnn.pack_padded_sequence(
            current, curr_lens_clamped, batch_first=True, enforce_sorted=False
        )
        
        # RNN encoding
        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            hidden_history_packed, _ = self.rnn_encoder(history_packed)
            hidden_state_packed, _ = self.rnn_decoder(current_packed)
        elif self.rnn_type == 'LSTM':
            hidden_history_packed, _ = self.rnn_encoder(history_packed)
            hidden_state_packed, _ = self.rnn_decoder(current_packed)
        
        # Unpack
        hidden_history, _ = torch.nn.utils.rnn.pad_packed_sequence(
            hidden_history_packed, batch_first=True
        )
        hidden_state, _ = torch.nn.utils.rnn.pad_packed_sequence(
            hidden_state_packed, batch_first=True
        )
        
        # Attention
        attn_weights = self.attn(hidden_state, hidden_history, hist_mask.float())
        context = torch.bmm(attn_weights, hidden_history)  # (batch, target_len, hidden)
        
        # Combine
        out = torch.cat([hidden_state, context], dim=2)  # (batch, target_len, 2*hidden)
        out = self.dropout(out)
        
        # Final projection
        y = self.fc_final(out)  # (batch, target_len, loc_size)
        score = F.log_softmax(y, dim=2)
        
        return score, curr_mask


class TrajectoryDataset(Dataset):
    """Dataset for trajectory prediction with variable length sequences"""
    
    def __init__(self, training_set, sos_token):
        logger.info("Creating TrajectoryDataset...")
        self.samples = []
        self.sos_token = sos_token  # Pass SOS token (model.loc_size)
        num_users = training_set['uid'].nunique()
        
        for uid, df_u in tqdm(training_set.groupby('uid'), total=num_users, desc="Processing users"):
            df_u = df_u.sort_values(['d', 't'])
            loc = ((df_u['x'] - 1) * 200 + (df_u['y'] - 1)).astype(int).values.tolist()
            tim = df_u['t'].astype(int).values.tolist()
            days = sorted(df_u['d'].unique().tolist())
            sessions = {}
            
            for d in days:
                idx = df_u['d'] == d
                loc_day = [loc[i] for i in range(len(loc)) if idx.iloc[i]]
                tim_day = [tim[i] for i in range(len(tim)) if idx.iloc[i]]
                
                # Deduplicate consecutive identical locations to reduce stationary bias
                dedup_loc = []
                dedup_tim = []
                if loc_day:
                    dedup_loc.append(loc_day[0])
                    dedup_tim.append(tim_day[0])
                    for i in range(1, len(loc_day)):
                        if loc_day[i] != dedup_loc[-1]:
                            dedup_loc.append(loc_day[i])
                            dedup_tim.append(tim_day[i])
                
                sessions[d] = (dedup_loc, dedup_tim)
            
            # Create training samples for this user
            for cur_d_idx in range(1, len(days)):
                cur_d = days[cur_d_idx]
                hist_days = days[:cur_d_idx]
                history_loc = sum((sessions[hd][0] for hd in hist_days), [])
                history_tim = sum((sessions[hd][1] for hd in hist_days), [])
                current_loc, current_tim = sessions[cur_d]
                target_len = len(current_loc)
                
                if target_len < 1:
                    continue
                
                # Add SOS at the start of current
                current_input_loc = [self.sos_token] + current_loc[:-1]
                current_input_tim = [current_tim[0]] + current_tim[1:]
                all_loc = history_loc + current_input_loc
                all_tim = history_tim + current_input_tim
                target = current_loc
                
                self.samples.append({
                    'uid': uid,
                    'loc': all_loc,
                    'tim': all_tim,
                    'target': target,
                    'target_len': target_len
                })
        
        logger.info(f"Created {len(self.samples)} training samples from {num_users} users")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'uid': sample['uid'],
            'loc': torch.LongTensor(sample['loc']),
            'tim': torch.LongTensor(sample['tim']),
            'target': torch.LongTensor(sample['target']),
            'target_len': sample['target_len']
        }


def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    # Extract fields
    uids = torch.LongTensor([item['uid'] for item in batch])
    target_lens = torch.LongTensor([item['target_len'] for item in batch])
    
    # Get sequence lengths for masking
    seq_lens = torch.LongTensor([len(item['loc']) for item in batch])
    
    # Pad sequences
    locs = rnn_utils.pad_sequence([item['loc'] for item in batch], batch_first=True, padding_value=0)
    tims = rnn_utils.pad_sequence([item['tim'] for item in batch], batch_first=True, padding_value=0)
    targets = rnn_utils.pad_sequence([item['target'] for item in batch], batch_first=True, padding_value=0)
    
    return {
        'uid': uids,
        'loc': locs,
        'tim': tims,
        'target': targets,
        'target_len': target_lens,
        'seq_len': seq_lens
    }
# Updated DeepmoveTrainer with batched training
class DeepmoveTrainer:
    def __init__(self, parameters):
        self.parameters = parameters
        self.model = TrajPreLocalAttnLong(parameters)
        if parameters.use_cuda:
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

    def train(self, training_set, epochs=10, batch_size=64, num_workers=0):
        from tqdm import tqdm
        import time
        
        # Create dataset and dataloader (pass SOS token)
        dataset = TrajectoryDataset(training_set, sos_token=self.model.sos_token)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            # num_workers=num_workers,
            # pin_memory=self.parameters.use_cuda
        )
        for epoch in range(epochs):
            epoch_start = time.time()
            self.model.train()
            
            total_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                # Move to device
                if self.parameters.use_cuda:
                    batch = {k: v.cuda() if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                # Forward pass on entire batch
                scores, mask = self.model(
                    batch['loc'], 
                    batch['tim'], 
                    batch['uid'],
                    batch['target_len'],
                    batch['seq_len']
                )
                
                # Calculate loss with masking
                scores_flat = scores.view(-1, scores.size(-1))
                targets_flat = batch['target'].view(-1)
                mask_flat = mask.view(-1)
                
                loss = F.nll_loss(scores_flat, targets_flat, reduction='none')
                loss = (loss * mask_flat.float()).sum() / mask_flat.float().sum()
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")


class DeepmoveForecaster:
    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters
        logger.info("Forecaster initialized")

    def forecast(self, training_set, predicting_days):
        logger.info(f"Starting forecasting for {predicting_days} days")

        # Preprocess user data
        logger.info("Preprocessing data for forecasting...")
        user_data = {}
        num_users = training_set['uid'].nunique()

        for uid, df_u in tqdm(training_set.groupby('uid'), total=num_users, desc="Processing users"):
            df_u = df_u.sort_values(['d', 't'])
            loc = ((df_u['x'] - 1) * 200 + (df_u['y'] - 1)).astype(int).values.tolist()
            tim = df_u['t'].astype(int).values.tolist()
            days = sorted(df_u['d'].unique().tolist())
            sessions = {}

            for d in days:
                idx = df_u['d'] == d
                loc_day = [loc[i] for i in range(len(loc)) if idx.iloc[i]]
                tim_day = [tim[i] for i in range(len(tim)) if idx.iloc[i]]
                
                # Deduplicate consecutive identical locations to reduce stationary bias
                dedup_loc = []
                dedup_tim = []
                if loc_day:
                    dedup_loc.append(loc_day[0])
                    dedup_tim.append(tim_day[0])
                    for i in range(1, len(loc_day)):
                        if loc_day[i] != dedup_loc[-1]:
                            dedup_loc.append(loc_day[i])
                            dedup_tim.append(tim_day[i])
                
                sessions[d] = (dedup_loc, dedup_tim)
            user_data[uid] = sessions

        logger.info(f"Preprocessed {len(user_data)} users for forecasting")

        pred_df = pd.DataFrame(columns=['uid', 'd', 't', 'x', 'y'])
        total_predictions = predicting_days * 48

        for uid in tqdm(user_data, desc="Forecasting users"):
            sessions = user_data[uid]
            days = sorted(sessions.keys())
            history_loc = sum((sessions[d][0] for d in days), [])
            history_tim = sum((sessions[d][1] for d in days), [])

            if not history_tim:
                logger.warning(f"User {uid} has no history, skipping")
                continue

            last_d = days[-1]
            last_t = history_tim[-1]
            current_t = (last_t + 1) % 48
            current_d = last_d if current_t != 0 else last_d + 1

            all_loc = history_loc[:]
            all_tim = history_tim[:]
            K = predicting_days * 48

            predicted_ds = []
            predicted_ts = []
            predicted_xs = []
            predicted_ys = []

            self.model.eval()
            logger.info(f"Forecasting {K} timesteps for user {uid}")

            with torch.no_grad():
                for step in tqdm(range(K), desc=f"Timesteps (uid={uid})", leave=False):
                    uid_tensor = torch.LongTensor([uid])
                    target_len_tensor = torch.LongTensor([1])
                    
                    # Handle SOS for start of new day
                    if current_t == 0:
                        temp_loc = all_loc + [self.model.sos_token]
                        temp_tim = all_tim + [current_t]
                    else:
                        temp_loc = all_loc
                        temp_tim = all_tim
                    
                    loc_tensor = torch.LongTensor(temp_loc)
                    tim_tensor = torch.LongTensor(temp_tim)
                    seq_len_tensor = torch.LongTensor([len(temp_loc)])

                    # Add batch dimension
                    loc_tensor = loc_tensor.unsqueeze(0)
                    tim_tensor = tim_tensor.unsqueeze(0)

                    if self.parameters.use_cuda:
                        loc_tensor = loc_tensor.cuda()
                        tim_tensor = tim_tensor.cuda()
                        uid_tensor = uid_tensor.cuda()
                        target_len_tensor = target_len_tensor.cuda()
                        seq_len_tensor = seq_len_tensor.cuda()

                    # Forward pass
                    score, _ = self.model(
                        loc_tensor,
                        tim_tensor,
                        uid_tensor,
                        target_len_tensor,
                        seq_len_tensor
                    )

                    # Use temperature sampling for more diverse predictions
                    temperature = 1.5  # Adjust for more/less diversity
                    probs = F.softmax(score[0, 0] / temperature, dim=0)
                    pred_loc = torch.multinomial(probs, 1).item()
                    
                    x = (pred_loc // 200) + 1
                    y = (pred_loc % 200) + 1

                    predicted_xs.append(x)
                    predicted_ys.append(y)
                    predicted_ds.append(current_d)
                    predicted_ts.append(current_t)

                    # Update sequence with predicted loc (not SOS)
                    all_loc.append(pred_loc)
                    all_tim.append(current_t)
                    current_t = (current_t + 1) % 48
                    if current_t == 0:
                        current_d += 1

            uid_list = [uid] * K
            df_user_pred = pd.DataFrame({
                'uid': uid_list,
                'd': predicted_ds,
                't': predicted_ts,
                'x': predicted_xs,
                'y': predicted_ys
            })
            pred_df = pd.concat([pred_df, df_user_pred], ignore_index=True)
            logger.info(f"Completed forecasting for user {uid}: {K} predictions")

        logger.info(f"Forecasting completed. Total predictions: {len(pred_df)}")
        return pred_df