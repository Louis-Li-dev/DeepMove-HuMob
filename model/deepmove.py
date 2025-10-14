# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Attn(nn.Module):
    """Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history):
        seq_len = history.size()[0]
        state_len = out_state.size()[0]
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = self.score(out_state[i], history[j])
        return F.softmax(attn_energies, dim=1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy


class TrajPreLocalAttnLong(nn.Module):
    """RNN model with long-term history attention, modified to include user embedding"""

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

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        self.emb_user = nn.Embedding(self.user_size, self.user_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size + self.user_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.RNN(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()
        logger.info(f"Model initialized: {self.rnn_type} with attention={self.attn_type}")

    def init_weights(self):
        """Here we reproduce Keras default initialization weights for consistency"""
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)
        logger.debug("Weights initialized successfully")

    def forward(self, loc, tim, target_len, uid):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        h2 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c2 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            h2 = h2.cuda()
            c1 = c1.cuda()
            c2 = c2.cuda()

        # Add batch dimension
        seq_len = loc.size(0)
        loc = loc.view(seq_len, 1)
        tim = tim.view(seq_len, 1)
        uid = uid.view(1, 1)

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        user_emb = self.emb_user(uid)
        user_emb = user_emb.repeat(seq_len, 1, 1)
        x = torch.cat((loc_emb, tim_emb, user_emb), 2)
        x = self.dropout(x)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            hidden_history, h1 = self.rnn_encoder(x[:-target_len], h1)
            hidden_state, h2 = self.rnn_decoder(x[-target_len:], h2)
        elif self.rnn_type == 'LSTM':
            hidden_history, (h1, c1) = self.rnn_encoder(x[:-target_len], (h1, c1))
            hidden_state, (h2, c2) = self.rnn_decoder(x[-target_len:], (h2, c2))

        hidden_history = hidden_history.squeeze(1)
        hidden_state = hidden_state.squeeze(1)
        attn_weights = self.attn(hidden_state, hidden_history).unsqueeze(0)
        context = attn_weights.bmm(hidden_history.unsqueeze(0)).squeeze(0)
        out = torch.cat((hidden_state, context), 1)
        out = self.dropout(out)

        y = self.fc_final(out)
        score = F.log_softmax(y, dim=1)

        return score


class DeepmoveTrainer:
    def __init__(self, parameters):
        self.parameters = parameters
        self.model = TrajPreLocalAttnLong(parameters)
        if parameters.use_cuda:
            self.model.cuda()
            logger.info("Model moved to GPU")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        logger.info("Trainer initialized with Adam optimizer")

    def train(self, training_set, epochs=10):
        logger.info(f"Starting training with {epochs} epochs")
        
        # Preprocess data into sessions per user per day
        logger.info("Preprocessing training data...")
        self.user_data = {}
        num_users = training_set['uid'].nunique()
        
        for uid, df_u in tqdm(training_set.groupby('uid'), total=num_users, desc="Processing users"):
            df_u = df_u.sort_values(['d', 't'])
            loc = ((df_u['x'] - 1) * 200 + (df_u['y'] - 1)).astype(int).values.tolist()
            tim = df_u['t'].astype(int).values.tolist()
            days = sorted(df_u['d'].unique().tolist())
            sessions = {}
            
            for d in days:
                idx = df_u['d'] == d
                sessions[d] = (
                    [loc[i] for i in range(len(loc)) if idx.iloc[i]],
                    [tim[i] for i in range(len(tim)) if idx.iloc[i]]
                )
            self.user_data[uid] = sessions
        
        logger.info(f"Preprocessed {len(self.user_data)} users")

        # Training loop
        for epoch in tqdm(range(epochs), desc="Epochs", position=0):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0.0
            count = 0
            
            for uid in tqdm(self.user_data, desc="Users", position=1, leave=False):
                sessions = self.user_data[uid]
                days = sorted(sessions.keys())
                
                for cur_d_idx in tqdm(range(1, len(days)), desc=f"Days (uid={uid})", position=2, leave=False):
                    cur_d = days[cur_d_idx]
                    hist_days = days[:cur_d_idx]
                    history_loc = sum((sessions[hd][0] for hd in hist_days), [])
                    history_tim = sum((sessions[hd][1] for hd in hist_days), [])
                    current_loc, current_tim = sessions[cur_d]
                    target_len = len(current_loc) - 1
                    
                    if target_len < 1:
                        continue
                    
                    all_loc = history_loc + current_loc[:-1]
                    all_tim = history_tim + current_tim[:-1]
                    
                    loc_tensor = torch.LongTensor(all_loc)
                    tim_tensor = torch.LongTensor(all_tim)
                    uid_tensor = torch.LongTensor([uid])
                    
                    if self.parameters.use_cuda:
                        loc_tensor = loc_tensor.cuda()
                        tim_tensor = tim_tensor.cuda()
                        uid_tensor = uid_tensor.cuda()
                    
                    self.model.train()
                    score = self.model(loc_tensor, tim_tensor, target_len, uid_tensor)
                    target = torch.LongTensor(current_loc[1:])
                    
                    if self.parameters.use_cuda:
                        target = target.cuda()
                    
                    loss = F.nll_loss(score, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    count += 1
            
            if count > 0:
                avg_loss = total_loss / count
                logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}, Batches: {count}")
            else:
                logger.warning(f"Epoch {epoch + 1}/{epochs}, No valid batches processed")
        
        logger.info("Training completed successfully")


class DeepmoveForecaster:
    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters
        logger.info("Forecaster initialized")

    def forecast(self, training_set, predicting_days):
        logger.info(f"Starting forecasting for {predicting_days} days")
        
        # Preprocess data similar to trainer
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
                sessions[d] = (
                    [loc[i] for i in range(len(loc)) if idx.iloc[i]],
                    [tim[i] for i in range(len(tim)) if idx.iloc[i]]
                )
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
            
            logger.info(f"Forecasting {total_predictions} timesteps for user {uid}")
            with torch.no_grad():
                for step in tqdm(range(K), desc=f"Timesteps (uid={uid})", leave=False):
                    loc_tensor = torch.LongTensor(all_loc)
                    tim_tensor = torch.LongTensor(all_tim)
                    uid_tensor = torch.LongTensor([uid])
                    
                    if self.parameters.use_cuda:
                        loc_tensor = loc_tensor.cuda()
                        tim_tensor = tim_tensor.cuda()
                        uid_tensor = uid_tensor.cuda()
                    
                    score = self.model(loc_tensor, tim_tensor, 1, uid_tensor)
                    pred_loc = torch.argmax(score[0]).item()
                    x = (pred_loc // 200) + 1
                    y = (pred_loc % 200) + 1
                    
                    predicted_xs.append(x)
                    predicted_ys.append(y)
                    predicted_ds.append(current_d)
                    predicted_ts.append(current_t)
                    
                    # Append for next prediction
                    all_loc.append(pred_loc)
                    next_tim = (current_t + 1) % 48
                    all_tim.append(next_tim)
                    current_t = next_tim
                    
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