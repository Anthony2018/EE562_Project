'''
Our model network
We modify this model from baseline model
https://github.com/YapengTian/AVE-ECCV18
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import pdb

def init_layers(layers):
    for layer in layers:
        nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0)


class attention_model(nn.Module):
    '''
    Self-attention-model: we use the self attenstion model for audio
    params:
    a_emb: auido_embbding size, 256
    h_sizeL inter hidden layer size, 128
    '''
    def __init__(self, a_emb, hzise=64):
        """
        phi and theta are the intergate threshold parameters
        """
        super(attention_model, self).__init__()
        self.phi = nn.Linear(a_emb, hzise)
        self.theta = nn.Linear(a_emb, hzise)
        self.g = nn.Linear(a_emb, hzise)
        att_layers = [self.phi, self.theta, self.g]
        init_layers(att_layers)

    def forward(self, fs):
        '''
        forward fucntion: forward function for self-attention
        params
        fs: audio_feature
        a_segemenate size is [batch_size, clip_size, clip_size]
        '''
        [batch, clip, a_emb] = fs.shape
        phi_a = self.phi(fs)
        theta_a = self.theta(fs)
        g_a = self.g(fs)
        a_segemenate = torch.bmm(phi_a, theta_a.permute(0, 2, 1))
        a_segemenate = a_segemenate / torch.sqrt(torch.FloatTensor([audio_emb_dim]).cuda())
        a_segemenate = F.relu(a_segemenate)
        a_segemenate = (a_segemenate + a_segemenate.permute(0, 2, 1)) / 2
        sum_a = torch.sum(a_segemenate, dim=-1, keepdim=True)
        a_segemenate = a_segemenate / (sum_a + 1e-8)
        audio_attention = torch.bmm(a_segemenate, g_a)
        out = audio_attention + fs
        return out, a_segemenate


class Audio_Visual_aug_model(nn.Module):
    """
    This part is modified from baselien model;
    https://github.com/YapengTian/AVE-ECCV18
    Audio-guided visual attention used in AVEL.
    AVEL:Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chen-liang Xu. Audio-visual event localization in unconstrained videos. InECCV, 2018
    """
    '''
    audio visual aug model:
    asize: audio feature size,128, from vgg model
    vsize: video feature size, 512, from vgg model
    hsize: hidden feature size, 512
    mapping size: can change, here we select 48;
    '''
    def __init__(self, asize=128, vsize=512, hsize=512, msize=48):
        super(Audio_Visual_aug_model, self).__init__()
        self.relu_layer = nn.ReLU()
        self.l_audio = nn.Linear(asize, hsize)
        self.l_video = nn.Linear(vsize, hsize)
        self.l_v = nn.Linear(hsize, msize, bias=False)
        self.l_g = nn.Linear(hsize, msize, bias=False)
        self.l_h = nn.Linear(msize, 1, bias=False)

        ## uniform initilization for affine funcitons: l_v,g,h,audio,video
        init.xavier_uniform(self.l_v.weight)
        init.xavier_uniform(self.l_g.weight)
        init.xavier_uniform(self.l_h.weight)
        init.xavier_uniform(self.l_audio.weight)
        init.xavier_uniform(self.l_video.weight)

    def forward(self, audio, video):
        video_size = video.size(-1)
        v_temp = video.view(video.size(0) * video.size(1), -1, video_size)
        V = v_temp

        v_temp = self.relu_layer(self.l_video(v_temp)) 
        a_temp = audio.view(-1, audio.size(-1)) 
        a_temp = self.relu_layer(self.l_audio(a_temp)) 
        inter = self.l_g(a_temp)
        content_v = self.l_v(v_temp) + inter.unsqueeze(2) 

        z_temp = self.l_h((F.tanh(content_v))).squeeze(2) 
        alpha_t = F.softmax(z_temp, dim=-1).view(z_temp.size(0), -1, z_temp.size(1)) 
        c_temp = torch.bmm(alpha_t, V).view(-1, vsize)
        video_temp = c_temp.view(video.size(0), -1, vsize)
        return video_temp


class LSTM_Audio_Visual(nn.Module):
    """
    This part is modified from baselien model;
    https://github.com/YapengTian/AVE-ECCV18
    Audio-guided visual attention used in AVEL.
    AVEL:Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chen-liang Xu. Audio-visual event localization in unconstrained videos. InECCV, 2018
    """

    '''
    LSTM_Audio_Visual: bi_lstm model
    asize: audio feature size
    vsize: video feature size
    hsize: hidden feature size, 128
    clip size: can change, here we select 10
    '''

    def __init__(self, asize, vsize, hzise=128, clip=10):
        super(LSTM_Audio_Visual, self).__init__()

        self.lstm_audio = nn.LSTM(asize, hzise, 1, batch_first=True, bidirectional=True, dropout=0.0)
        self.lstm_video = nn.LSTM(vsize, hzise, 1, batch_first=True, bidirectional=True, dropout=0.0)

    def init_hidden(self, a_f, v_f):
        batch, clip, asize = a_f.shape
        hidden_a = (torch.zeros(2, batch, asize).cuda(), torch.zeros(2, batch, asize).cuda())
        hidden_v = (torch.zeros(2, batch, asize).cuda(), torch.zeros(2, batch, asize).cuda())
        return hidden_a, hidden_v

    def forward(self, a_f, v_f):
        hidden_a, hidden_v = self.init_hidden(a_f, v_f)
        # Bi-LSTM for temporal modeling
        self.lstm_video.flatten_parameters()
        self.lstm_audio.flatten_parameters()
        lstm_audio, hidden1 = self.lstm_audio(a_f, hidden_a)
        lstm_video, hidden2 = self.lstm_video(v_f, hidden_v)
        return lstm_audio, lstm_video


class our_model(nn.Module):
    """our module
    asize: the size of audio feature, it is from vgg cnn output
    vsize: the size of video feature, ti is the output of vgg cnn output
    hsize: hiddent layer size, 256
    out_dim: out put size, 256
    """

    def __init__(self, asize=256, vsize=256, hzise=256, out_dim=256):
        super(our_model, self).__init__()
        self.v_1 = nn.Linear(vsize, hzise, bias=False)
        self.v_2 = nn.Linear(vsize, hzise, bias=False)
        self.vfc = nn.Linear(vsize, out_dim, bias=False)
        self.a_1 = nn.Linear(asize, hzise, bias=False)
        self.a_2 = nn.Linear(asize, hzise, bias=False)
        self.a_c = nn.Linear(asize, out_dim, bias=False)
        self.relu_layer = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1) # default=0.1
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-6)

        layers = [self.v_1, self.v_2, self.a_1, self.a_2, self.afc, self.vfc]
        self.init_weights(layers)

    def init_weights(self, layers):
        for layer in layers:
            nn.init.xavier_uniform(layer.weight)

    def forward(self, a_f, v_f, threshold):
        v_b1 = self.dropout(self.relu_layer(self.v_1(v_f))) #[batch, 10, hzise], 10 is clip seconds
        v_b2 = self.dropout(self.relu_layer(self.v_2(v_f)))
        a_b1 = self.dropout(self.relu_layer(self.a_1(a_f)))
        a_b2 = self.dropout(self.relu_layer(self.a_2(a_f)))
        # beta
        beta_v_a = torch.bmm(v_b2, a_b1.permute(0, 2, 1))
        beta_v_a /= torch.sqrt(torch.FloatTensor([v_b2.shape[2]]).cuda())

        beta_v_a = self.relu_layer(beta_v_a) # ReLU
        beta_a_v = beta_v_a.permute(0, 2, 1) # transpose

        sum_v_a = torch.sum(beta_v_a, dim=-1, keepdim=True)
        beta_v_a = beta_v_a / (sum_v_a + 1e-8)


        # gamma
        gamma_v_a = (beta_v_a > threshold).float() * beta_v_a
        sum_v_a = torch.sum(gamma_v_a, dim=-1, keepdim=True)
        gamma_v_a = gamma_va / (sum_v_a + 1e-8)

        sum_a_to_v = torch.sum(beta_a_v, dim=-1, keepdim=True)
        beta_a_v = beta_a_v / (sum_a_to_v + 1e-8)
        gamma_a_v = (beta_a_v > threshold).float() * beta_a_v
        sum_a_to_v = torch.sum(gamma_a_v, dim=-1, keepdim=True)
        gamma_a_v = gamma_a_v / (sum_a_to_v + 1e-8)
        # pos == postive 

        a_pos = torch.bmm(gamma_v_a, a_b2)
        v_out = v_fea + a_pos
        v_pos = torch.bmm(gamma_a_v, v_b1)
        a_out = a_fea + v_pos
        ## affine the v_out and a_out
        v_out = self.dropout(self.relu_layer(self.v_fc(v_out)))
        a_out = self.dropout(self.relu_layer(self.a_fc(a_out)))
        v_out = self.layer_norm(v_out)
        a_out = self.layer_norm(a_out)

        av_f = torch.mul(v_out + a_out, 0.4)
        return av_f, v_out, a_out



class Classifier(nn.Module):
    # last layer classificy
    # 28 classes, 1 is background, not inclused here
    def __init__(self, hzise=256, class_num=28):
        super(Classifier, self).__init__()
        self.L1 = nn.Linear(hzise, 128, bias=False)
        self.L2 = nn.Linear(128, class_num, bias=False)
    def forward(self, feature):
        out = F.relu(self.L1(feature))
        out = self.L2(out)
        return out


class Similarity(nn.Module):
    """ function to compute audio-visual similarity
        Cosine Similarity
    """
    def __init__(self,):
        super(Similarity, self).__init__()

    def forward(self, v_f, a_f):
        v_f = F.normalize(v_f, dim=-1)
        a_f = F.normalize(a_f, dim=-1)
        cos = torch.sum(torch.mul(v_f, a_f), dim=-1) # [batch, 10]
        return cos



class fully_supervised(nn.Module):
    '''
    System flow for fully supervised audio-visual event localization.
    asize: audio feature size
    vsize: video feature size
    hsize: hidden feature size, 128
    class_num: 28 classes, 1 is background
    '''
    def __init__(self, asize=128, vsize=512, hzise=128, class_num=29):
        super(fully_supervised, self).__init__()
        self.function_a = nn.Sequential(
            nn.Linear(asize, 256, bias=False),
            nn.Linear(256, 128, bias=False),
        )
        self.function_v = nn.Sequential(
            nn.Linear(vsize, 256, bias=False),
            nn.Linear(256, 128, bias=False),
        )
        self.affine_v = nn.Linear(vsize, asize)
        self.relu_layer = nn.ReLU()
        self.attention = Audio_Visual_aug_model(vsize=vsize)
        self.lstm_a_v = LSTM_Audio_Visual(asize=asize, vsize=hzise, hzise=hzise)
        self.model = our_model(asize=asize*2, vsize=hzise*2)
        self.av_simm = Similarity()

        self.v_classifier = Classifier(hzise=256)
        self.a_classifier = Classifier(hzise=256)

        self.L1 = nn.Linear(2*hzise, 64, bias=False)
        self.L2 = nn.Linear(64, class_num, bias=False)

        self.L3 = nn.Linear(256, 64)
        self.L4 = nn.Linear(64, 2)
        # layers = [self.L1, self.L2]
        layers = [self.L1, self.L2, self.L3, self.L4]
        self.init_layers(layers)

    def init_layers(self, layers):
        for layer in layers:
            nn.init.xavier_uniform(layer.weight)

    def forward(self, audio, video, threshold):
        batch, clip, _, _, vsize = video.shape
        function_a_a = self.function_a(audio)
        video_t = self.attention(function_a_a, video) # [batch, 10, 512]
        video_t = self.function_v(video_t) # [batch, 10, 128]
        lstm_audio, lstm_video = self.lstm_a_v(function_a_a, video_t)
        fusion, final_v_f, final_a_f = self.psp(lstm_audio, lstm_video, threshold=threshold) # [batch, 10, 256]
        cross_att = self.av_simm(final_v_f, final_a_f)

        out = self.relu_layer(self.L1(fusion))
        pred = self.L2(out) # [batch, 10, 29]
        return fusion, pred, cross_att

