import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.features = configs.features

        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=self.dropout if configs.e_layers > 1 else 0,
        )

        # Đầu ra dự đoán
        self.fc = nn.Linear(self.d_model, self.pred_len * self.enc_in)
        self.dropout = nn.Dropout(self.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc: [batch_size, seq_len, enc_in]
        x_mark_enc: [batch_size, seq_len, num_time_features]
        """
        # Khởi tạo trạng thái ẩn
        batch_size = x_enc.size(0)
        device = x_enc.device

        # LSTM forward
        output, (hn, cn) = self.lstm(x_enc)  # output: [batch_size, seq_len, d_model]

        # Lấy output cuối cùng
        last_output = output[:, -1, :]  # [batch_size, d_model]

        # Áp dụng dropout và fully connected layer
        last_output = self.dropout(last_output)
        dec_out = self.fc(last_output)  # [batch_size, pred_len * enc_in]

        # Reshape output
        if self.features == "M" or self.features == "MS":
            dec_out = dec_out.view(batch_size, self.pred_len, self.enc_in)
        else:
            dec_out = dec_out.view(batch_size, self.pred_len, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [batch_size, pred_len, enc_in]
        return None
