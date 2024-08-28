import pickle
import torch
import numpy as np
from user import user
from server import server
import math
import argparse
import warnings
import logging
import faulthandler
faulthandler.enable()
warnings.filterwarnings('ignore')

logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

con = 0.6

nei = 0.4
logger.info(f" nei = {nei}, con = {con}")


def set_random_seed(seed, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="args for FedGNN")
parser.add_argument('--embed_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--data', default='filmtrust')
parser.add_argument('--user_batch', type=int, default=256)
parser.add_argument('--clip', type=float, default=0.3)
parser.add_argument('--laplace_lambda', type=float, default=0.1)
parser.add_argument('--negative_sample', type=int, default=10)
parser.add_argument('--valid_step', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--con_rate', type=float, default=0.001)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

embed_size = args.embed_size
user_batch = args.user_batch
lr = args.lr
device = torch.device('cuda:0')
if torch.cuda.is_available():
    device = torch.device("cuda")  # 选择第一个可用的GPU设备


def processing_valid_data(valid_data):
    res = []
    for key in valid_data.keys():
        if len(valid_data[key]) > 0:
            for ratings in valid_data[key]:
                item, rate, _ = ratings
                res.append((int(key), int(item), rate))
                """res相当于把验证集中的数据重新处理了一遍，得到的res为一个列表：（原验证集的条数（key，可能这个是用户号），项目号，评分）"""
    return np.array(res)


def loss(server, valid_data):
    label = valid_data[:, -1]
    predicted = server.predict(valid_data)
    mae = sum(abs(label - predicted)) / len(label)
    rmse = math.sqrt(sum((label - predicted) ** 2) / len(label))
    return mae, rmse


# read data
data_file = open('../data/' + args.data + '_FedMF.pkl', 'rb')
[train_data, valid_data, test_data, user_id_list, item_id_list, social] = pickle.load(data_file)
data_file.close()
valid_data = processing_valid_data(valid_data)
test_data = processing_valid_data(test_data)

# build user_list
rating_max = -9999
rating_min = 9999
user_list = []
for u in user_id_list:
    all_number = train_data[u]
    items = []
    rating = []
    for i in range(len(all_number)):
        item, rate, _ = all_number[i]
        items.append(item)
        rating.append(rate)

    pos_neg_list = []
    if len(rating) > 1:
        mean_rating = sum(rating) / len(rating)
        for i in range(len(rating)):
            if rating[i] > mean_rating:
                pos_neg_list.append(1)
            else:
                pos_neg_list.append(0)

        if all(element == 0 for element in pos_neg_list):
            pos_neg_list = []

        rating_max = max(rating_max, max(rating))
        rating_min = min(rating_min, min(rating))
    user_list.append(
        user(u, items, rating, list(social[u]), pos_neg_list, embed_size, args.clip, args.laplace_lambda,
             args.negative_sample, nei, con))

# build server
server = server(user_list, user_batch, user_id_list, item_id_list, embed_size, lr, device, rating_max, rating_min,
                args.weight_decay)
count = 0

# train and evaluate
rmse_best = 9999
while 1:
    for i in range(args.valid_step):
        server.train()
    logger.info('valid')
    mae, rmse = loss(server, valid_data)
    logger.info('valid mae: {}, valid rmse:{}'.format(mae, rmse))
    if rmse < rmse_best:
        rmse_best = rmse
        count = 0
        mae_test, rmse_test = loss(server, test_data)
    else:
        count += 1
    if count > 5:
        print('not improved for 5 epochs, stop trianing')
        break
logger.info('final test mae: {}, test rmse: {}'.format(mae_test, rmse_test))




