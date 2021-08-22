from __future__ import print_function
import h5py
import pickle
import random
import matplotlib.pyplot as plt
import argparse
import os
from DataLoader import DataLoader_Transform, DataLoader_Transform_Single_Scan, DataLoader_Transform_TTA, DataLoader_Transform_Upright

import torch

from model import ActorCritic, command2action
from tensorboardX import SummaryWriter
import datetime
import numpy as np
import logging
import torch.nn.functional as F

import trans
import matplotlib
matplotlib.use("Agg")
plt.rcParams['figure.figsize'] = (8.0, 4.0)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def str2bool(v):
      return v.lower() in ('true', '1')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='A2C')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.4,
                        help='discount factor for rewards (default: 0.4)')
    # parser.add_argument('--gae-lambda', type=float, default=1.00, help='lambda parameter for GAE (default: 1.00)')
    parser.add_argument('--entropy_coef', type=float, default=0.1,
                        help='entropy term coefficient (default: 0.1)')
    parser.add_argument('--value_loss_coef', type=float,
                        default=1, help='value loss coefficient (default: 1)')
    parser.add_argument('--policy_loss_coef', type=float,
                        default=1, help='value loss coefficient (default: 1)')
    parser.add_argument('--max_grad_norm', type=float,
                        default=50, help='value loss coefficient (default: 50)')
    parser.add_argument('--clone', type=str2bool, default=False)
    parser.add_argument("--use_tensorboard", type=bool, default=True)
    parser.add_argument('--decay_rate', type=float,default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--loss', type=str, default='l2' ) # l2, cos, acos
    
    parser.add_argument("--n_epochs", type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--task', type=str, default='test', help='train / test' ) # train, test
    parser.add_argument("--hidden_dim", type=int, default=128)

    parser.add_argument('--num_steps', type=int, default=20, help='Number of forward steps in A3C (default: 20)')
    parser.add_argument('--rot_angle', type=int, default=4 )
    parser.add_argument('--action_num', type=int, default=5, help='The output actions num')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 1024]')

    parser.add_argument('--data_type', type=str, default='single_scan', help='complete / partial / single_scan / uprl' ) # complete, partial, single_scan, uprl
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Dataset folder' )

    parser.add_argument('--result_dir', type=str, default='../test_output', help='Save result folder')
    parser.add_argument('--pretrain_dir', type=str, default=None, help='Pretrain model path')

    parser.add_argument('--note', type=str, default='debug', help='Additional comments')

    return parser.parse_args()


step_epoch = 0
best_axis = 1000
best_angle = 1000
best_epoch_ = 1000
best_origin_point = []
best_gt_para_r = []
best_out_para_r = []
best_out_point = []
best_accu = 0
_EPS = np.finfo(float).eps * 4.0

def log_string(logger, str):
    logger.info(str)
    print(str)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_reward_batch_with_stop(batch_size, previou, current, t, terminal, loss_type):
    batch_size = len(previou)
    reward = torch.zeros(batch_size)
    for i in range(batch_size):
        if terminal[i] == 0:
            if loss_type == 'cos':
                if current[i] > previou[i]:
                    reward[i] = 5 - 0.001*t
                else:
                    reward[i] = -1 - 0.001*t
            else:
                if current[i] < previou[i]:
                    reward[i] = 5 - 0.001*t
                else:
                    reward[i] = -1 - 0.001*t
        else:
            reward[i] = 0
    return reward

def plot_and_save(path_name, data, xlabel, ylabel, title  ):
    with open(path_name, "wb") as file:
        pickle.dump(data, file)
    x = np.arange(1, len(data) + 1)

    # plot the val loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(x, data, "r-")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(fontsize=8)
    path_name = path_name[:-3] + 'png'
    plt.savefig(path_name)
    plt.close(1)

def main(args):

    def train(args, mode, dataloader, model, optimizer, epoch, logger, summary_writer=None):
        
        error_axis_l2_sum, error_axis_rad_sum = [], []
        model = model.train()

        policy_loss_epoch = []
        value_loss_epoch = []
        total_rewards_epoch = []

        for batch_id, (point_input, gt_up) in enumerate(dataloader):

            optimizer.zero_grad()

            origin_point, gt_para_r = point_input, gt_up
            origin_point, gt_para_r = origin_point.float().cuda(), gt_para_r.float().cuda()

            # tmp_origin = origin_point.clone()

            entropies = torch.zeros(args.num_steps, args.batch_size)
            values = torch.zeros(args.num_steps, args.batch_size)
            log_probs = torch.zeros(args.num_steps, args.batch_size)
            rewards = torch.zeros(args.num_steps, args.batch_size)

            compose_matrix = np.identity(4)
            compose_matrix_b = np.repeat(np.expand_dims(
                compose_matrix, axis=0), args.batch_size, axis=0)
                
            # initialize the origin upright
            origin_para_r0 = torch.tensor([[0., 0., 1.]]).float()
            origin_para_r0 = origin_para_r0.repeat( (args.batch_size, 1) ).cuda()

            # store num_step that the point needn't transform
            terminal_index = np.repeat([0], args.batch_size, axis=0)

            origin_score = []
            # 开始网络执行 
            # start rotate
            for step in range(args.num_steps):
                # initialize paras(paras store the angle and trans_axis)
                paras = np.repeat([[0, 0, 0, 0]], args.batch_size, axis=0)
                
                if step == 0:
                    hx = torch.zeros(args.batch_size, 128).cuda()
                    cx = torch.zeros(args.batch_size, 128).cuda()
                    
                    if args.clone:
                        observation_ts = origin_point.clone()
                        observation_para = origin_para_r0.clone()
                    else:
                        observation_ts = origin_point
                        observation_para = origin_para_r0

                    for i in range(args.batch_size):
                        if args.loss == 'l2':
                            origin_score.append(F.mse_loss(origin_para_r0[i], gt_para_r[i]))

                        elif args.loss == 'cos':
                            obs_para = observation_para[i].cpu().detach().numpy()
                            gt_p = gt_para_r.cpu().detach().numpy()[i]
                            temp_dot = np.dot(obs_para, gt_p)/(np.linalg.norm(obs_para) * np.linalg.norm(gt_p)+_EPS) 
                            if temp_dot < -1:
                                temp_dot = -1
                            elif temp_dot > 1:
                                temp_dot = 1
                            origin_score.append(temp_dot)
                        
                        elif args.loss == 'acos':
                            temp_dot = np.dot(observation_para[i].cpu().detach().numpy(), gt_para_r.cpu().detach().numpy()[i])
                            if temp_dot < -1:
                                temp_dot = -1
                            elif temp_dot > 1:
                                temp_dot = 1
                            origin_score.append(np.abs(np.arccos(temp_dot)))

                    current_score = origin_score

                # network step
                observation_to = observation_ts.type(torch.FloatTensor).cuda()
                value, logit, (hx, cx) = model((observation_to, (hx, cx)))

                # get the probability of actions
                prob = F.softmax(logit, dim=1)
                log_prob = F.log_softmax(logit, dim=1)
                entropy = -(log_prob * prob).sum(1)
                entropies[step, :] = entropy

                # sample an action (index)
                action = prob.multinomial(num_samples=1).data
                log_prob = log_prob.gather(1, action)
                action = action.cpu().numpy()[:, 0]
                # action index to actual rotation action
                paras, terminal_index = command2action(action, paras, terminal_index, step, args.rot_angle)

                score = []

                # transform the point cloud by action
                for i in range(args.batch_size):
                    angle = paras[i, 0] * np.pi / 180.0
                    axis = paras[i, 1:]
                    if terminal_index[i] == 0:
                        rotation_matrix = trans.axisangle2matrix(angle, axis)
                        
                        # compose_matrix_b 是累积物体变换的矩阵,表示从z-axis正方向转到物体实际朝向的矩阵
                        # accumulate the transform, compose_matrix_b is the transform from upright z-axis to current direction
                        compose_matrix_b[i] = np.dot( rotation_matrix, compose_matrix_b[i])
                        # 对初始物体点云进行逆变换,靠近摆正的状态
                        # transform the orginal point cloud to close to upright state
                        observation_ts[i] = trans.transform_pts_torch( origin_point[i], torch.from_numpy(compose_matrix_b[i].T).cuda())
                        # 对正方向进行变换,逐渐靠近物体实际朝向
                        # transform the upright direction to close to object direction
                        observation_para[i] = trans.transform_pts_torch( origin_para_r0[i], torch.from_numpy(compose_matrix_b[i]).cuda())

                    
                    # 计算当前朝向和gt的差异，用于后面计算reward
                    if args.loss == 'l2':
                        next_score = F.mse_loss(observation_para[i], gt_para_r[i])

                    elif args.loss == 'cos':

                        obs_para = observation_para[i].cpu().detach().numpy()
                        gt_p = gt_para_r.cpu().detach().numpy()[i]
                        temp_dot = np.dot(obs_para, gt_p)/(np.linalg.norm(obs_para) * np.linalg.norm(gt_p)+_EPS)

                        if temp_dot < -1:
                            temp_dot = -1
                        elif temp_dot > 1:
                            temp_dot = 1
                        next_score = temp_dot

                    elif args.loss == 'acos':
                        temp_dot = np.dot(observation_para[i].cpu().detach().numpy(), gt_para_r.cpu().detach().numpy()[i])
                        
                        if temp_dot < -1:
                            temp_dot = -1
                        elif temp_dot > 1:
                            temp_dot = 1
                        next_score = np.abs(np.arccos(temp_dot))

                    score.append(next_score)

                reward = calculate_reward_batch_with_stop( args.batch_size, current_score, score, step+1, terminal_index, args.loss)

                current_score = score
                values[step, :] = value.squeeze(1)
                log_probs[step, :] = log_prob.squeeze(1)
                rewards[step, :] = reward

                total_rewards_epoch.append(rewards.sum())

            policy_loss = 0
            value_loss = 0
            idx = 0
            
            for j in range(args.batch_size):
                if terminal_index[j] == 0:
                    terminal_index[j] = args.num_steps
                    
            for j in range(args.batch_size):
                for k in reversed(range(terminal_index[j])):
                    if k == terminal_index[j]-1:
                        R = args.gamma * values[k][j] + rewards[k][j]
                    else:
                        R = args.gamma * R + rewards[k][j]

                    advantage = R - values[k][j]

                    value_loss = value_loss + advantage.pow(2)
                    policy_loss = policy_loss - log_probs[k][j] * advantage - args.entropy_coef * entropies[k][j]
                    idx += 1

            policy_loss /= idx
            value_loss /= idx

            policy_loss_epoch.append(policy_loss.data)
            value_loss_epoch.append(value_loss.data)

            (args.policy_loss_coef * policy_loss + args.value_loss_coef * value_loss).backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            err_axis_l2, err_axis_rad, err_angle = 0., 0., 0.
            for i in range(args.batch_size):
                err_axis_l2 += F.mse_loss(observation_para[i], gt_para_r[i])
                err_axis_rad += np.abs(np.arccos(np.dot(observation_para[i].cpu().numpy(), gt_para_r.cpu().detach().numpy()[i])/(
                    np.linalg.norm(observation_para[i].cpu().numpy()) * np.linalg.norm(gt_para_r.cpu().detach().numpy()[i])+_EPS)))

            err_axis_l2 /= args.batch_size
            err_axis_rad /= args.batch_size
            error_axis_l2_sum.append(err_axis_l2.item())
            error_axis_rad_sum.append(err_axis_rad)

            # Batch log
            global step_epoch
            if summary_writer:
                summary_writer.add_scalar('loss_policy', policy_loss.data, step_epoch)
                summary_writer.add_scalar('loss_value', value_loss.data, step_epoch)
                summary_writer.add_scalar('loss_total', 
                    args.policy_loss_coef * policy_loss.data+args.value_loss_coef * value_loss.data, step_epoch)
            step_epoch += 1

        log_string(logger, mode+' error axis l2: %f' % np.mean(error_axis_l2_sum))
        log_string(logger, mode+' error axis rad: %f' % np.mean(error_axis_rad_sum))

        ave_policy_loss = sum(policy_loss_epoch) / len(policy_loss_epoch)
        ave_value_loss = sum(value_loss_epoch) / len(value_loss_epoch)
        ave_total_rewards_epoch = sum(total_rewards_epoch) / len(total_rewards_epoch)
        loss_total = args.policy_loss_coef * ave_policy_loss + args.value_loss_coef * ave_value_loss

        print("Average Policy Loss for Train Epoch %d : %f" % (epoch+1, ave_policy_loss))
        print("Average Value Loss for Train Epoch %d : %f" % (epoch+1, ave_value_loss))
        print("Average Total Reward for Train Epoch %d : %f" % (epoch+1, ave_total_rewards_epoch))

        log_string(logger, "\nEpoch [%2d/%2d] : Tot Loss: %5.5f, Tot Rewards: %5.5f, Policy Loss: %5.5f, Value Loss: %5.5f" %
                   (epoch+1, args.n_epochs, loss_total, ave_total_rewards_epoch, ave_policy_loss, ave_value_loss))
 
        ave_train_policy_loss_all.append(ave_policy_loss)
        ave_train_value_loss_all.append(ave_value_loss)
        ave_train_total_rewards_all.append(ave_total_rewards_epoch)
        loss_total_all.append(loss_total)

        ave_policy_loss_all = ave_train_policy_loss_all
        ave_value_loss_all = ave_train_value_loss_all
        ave_total_rewards_all = ave_train_total_rewards_all

        err_l2 = np.mean(error_axis_l2_sum)
        err_rad = np.mean(error_axis_rad_sum)
        ave_train_err_axis_all.append(err_l2)
        ave_train_err_angle_all.append(err_rad)

        # 绘制折线图
        if True:
            plot_and_save(plot_dir + "/" + mode + "_" + args.note+"_iteration_ave_reward.pkl", 
                ave_total_rewards_all, "Iteration", 'Rewards', "Average Reward iteration")
            
            plot_and_save(plot_dir + "/" + mode + "_" + args.note+"_iteration_ave_total_loss.pkl", 
                loss_total_all, "Iteration", 'Loss', "Average Total Loss iteration")
            
            plot_and_save(plot_dir + "/" + mode + "_" + args.note+"_iteration_ave_policy_loss.pkl", 
                ave_policy_loss_all, "Iteration", 'Loss', "Average Policy Loss iteration")
            
            plot_and_save(plot_dir + "/" + mode + "_" + args.note+"_iteration_ave_value_loss.pkl", 
                ave_value_loss_all, "Iteration", 'Loss', "Average Value Loss iteration")
            
            plot_and_save(plot_dir + "/" + mode + "_" + args.note+"_iteration_ave_axis_l2.pkl", 
                ave_train_err_axis_all, "epoch", 'err_axis', "Average err_axis iteration")
            
            plot_and_save(plot_dir + "/" + mode + "_" + args.note+"_iteration_ave_axis_rad.pkl", 
                ave_train_err_angle_all, "epoch", 'err_axis', "Average err_axis iteration")

        return loss_total

    def test(args, mode, dataloader, model, epoch):
        global best_axis
        global best_angle
        global best_epoch_
        global best_origin_point
        global best_gt_para_r
        global best_out_para_r
        global best_out_point
        global best_accu

        error_axis_l2_sum, error_axis_rad_sum = [], []
        model = model.eval()
        accuracy_15 = 0
        accuracy_10 = 0
        accuracy_5 = 0

        temp_out_para_r = []
        temp_out_point = []

        for batch_id, (point_input, gt_up) in enumerate(dataloader):

            origin_point, gt_para_r = point_input, gt_up
            origin_point, gt_para_r = origin_point.float().cuda(), gt_para_r.float().cuda()

            compose_matrix = np.identity(4)
            compose_matrix_b = np.repeat(np.expand_dims(
                compose_matrix, axis=0), args.batch_size, axis=0)

            origin_para_r0 = torch.tensor([[0., 0., 1.]]).float()
            origin_para_r0 = origin_para_r0.repeat( (args.batch_size, 1) ).cuda()

            # store num_step that the point needn't transform
            terminal_index = np.repeat([0], args.batch_size, axis=0)
            
            for step in range(args.num_steps):

                paras = np.repeat([[0, 0, 0, 0]], args.batch_size, axis=0)

                if step == 0:
                    hx = torch.zeros(args.batch_size, 128).cuda()
                    cx = torch.zeros(args.batch_size, 128).cuda()

                    if args.clone:
                        observation_ts = origin_point.clone()
                        observation_para = origin_para_r0.clone()
                    else:
                        observation_ts = origin_point
                        observation_para = origin_para_r0

                observation_to = observation_ts.type(torch.FloatTensor).cuda()
                value, logit, (hx, cx) = model((observation_to, (hx, cx)))

                prob = F.softmax(logit, dim=1)
                # charades
                action = prob.max(1, keepdim=True)[1].data.cpu().numpy()[:, 0]
                # 将action转为角度变化
                paras, terminal_index = command2action(action, paras, terminal_index, step, args.rot_angle)


                # 执行变换
                for i in range(args.batch_size):
                    angle = paras[i, 0] * np.pi/180.0
                    axis = paras[i, 1:]
                    if terminal_index[i] == 0:
                        rotation_matrix = trans.axisangle2matrix(angle, axis)

                        # 累计变换矩阵，这里的变换矩阵是指从 正方向转到实际物体方向 的变换
                        # accumulate the transform, compose_matrix_b is the transform from upright to current direction
                        compose_matrix_b[i] = np.dot(rotation_matrix, compose_matrix_b[i])
                        
                        # 对初始物体点云进行逆变换,靠近摆正的状态
                        # transform the orginal point cloud to close to upright state
                        observation_ts[i] = trans.transform_pts_torch( origin_point[i], torch.from_numpy(compose_matrix_b[i].T).cuda())
                        # 对正方向进行变换,逐渐靠近物体实际朝向
                        # transform the upright direction to close to object direction
                        observation_para[i] = trans.transform_pts_torch( origin_para_r0[i], torch.from_numpy(compose_matrix_b[i]).cuda())

                        # 对输入点云进行逆变换，不断摆正
                        # observation_ts[i] = trans.transform_pts_torch(
                        #     origin_point[i].cpu(), torch.from_numpy(compose_matrix_b[i].T))
                        # # 对z-axis进行变换，不断靠近实际朝向
                        # observation_para[i] = trans.transform_pts(origin_para_r0[i], compose_matrix_b[i])

            err_axis_l2, err_axis_rad, err_angle = 0., 0., 0.
            for i in range(args.batch_size):
                err_axis_l2 += F.mse_loss(observation_para[i], gt_para_r[i])

                obs_para = observation_para[i].cpu().numpy()
                error_i = np.abs(np.arccos(np.dot(obs_para, gt_para_r.cpu().detach().numpy()[
                                 i])/(np.linalg.norm(obs_para) * np.linalg.norm(gt_para_r.cpu().detach().numpy()[i])+_EPS)))
                                 
                err_axis_rad += error_i
                if error_i < 15.0 * np.pi/180.0:
                    accuracy_15 += 1
                    if error_i < 10.0 * np.pi/180.0:
                        accuracy_10 += 1
                        if error_i < 5.0 * np.pi/180.0:
                            accuracy_5 += 1
                            
            err_axis_l2 /= args.batch_size
            err_axis_rad /= args.batch_size
            error_axis_l2_sum.append(err_axis_l2.item())
            error_axis_rad_sum.append(err_axis_rad)

            if epoch == 0:
                best_origin_point.append(point_input.cpu().detach().numpy())
                best_gt_para_r.append(gt_up.cpu().detach().numpy())

            temp_out_para_r.append(observation_para.cpu().numpy())
            temp_out_point.append(observation_ts.cpu().numpy())

        data_len = len(TEST_DATASET)

        log_string(logger, mode + ' error axis l2: %f' % np.mean(error_axis_l2_sum))
        log_string(logger, mode + ' error axis rad: %f' % np.mean(error_axis_rad_sum))
        # log_string(logger,'%s acc_15: %f' % (mode, accuracy_15 / data_len))
        # log_string(logger,'%s acc_10: %f' % (mode, accuracy_10 / data_len))
        # log_string(logger,'%s acc_05: %f' % (mode, accuracy_5 / data_len))
    
        real_test_len = (len(dataloader) * args.batch_size)
        # log_string(logger,"Because of dop_last is true,the real number of test data used in test is: "+str(real_test_len))
        log_string(logger, " ")
        log_string(logger, '%s acc_15: %f' % (mode, accuracy_15 / real_test_len))
        log_string(logger, '%s acc_10: %f' % (mode, accuracy_10 / real_test_len))
        log_string(logger, '%s acc_05: %f' % (mode, accuracy_5 / real_test_len))

        err_l2 = np.mean(error_axis_l2_sum)
        err_rad = np.mean(error_axis_rad_sum)
        ave_err_axis_all.append(err_l2)
        ave_err_angle_all.append(err_rad)
        acc_15_all.append(accuracy_15 / data_len)
        acc_10_all.append(accuracy_10 / data_len)
        acc_5_all.append(accuracy_5 / data_len)
        accu =  1.0 * accuracy_15 / data_len

        if True:
            plot_and_save(plot_dir + "/" + mode + "_iteration_acc_15.pkl", 
                acc_15_all, "epoch", 'acc_15', "acc iteration")
            plot_and_save(plot_dir + "/" + mode + "_iteration_acc_10.pkl", 
                acc_10_all, "epoch", 'acc_10', "acc iteration")
            plot_and_save(plot_dir + "/" + mode + "_iteration_acc_5.pkl", 
                acc_5_all, "epoch", 'acc_5', "acc iteration")
            plot_and_save(plot_dir + "/" + mode + "_iteration_ave_axis_l2.pkl", 
                ave_err_axis_all, "epoch", 'err_axis', "Average err_axis iteration")
            plot_and_save(plot_dir + "/" + mode + "_iteration_ave_axis_rad.pkl", 
                ave_err_angle_all, "epoch", 'err_rad', "Average err_rad iteration")

        if accu > best_accu or err_l2 < best_axis or err_rad < best_angle:
            best_axis = err_l2
            best_angle = err_rad
            best_accu = accu

            best_epoch_ = epoch + 1

            # log_string(logger, "best err_axis_l2: %0.5f" % best_axis)
            # log_string(logger, "best err_axis_rad: %0.5f" % best_angle)
            # log_string(logger, "epoch: %d" % best_epoch_)

            os.makedirs(str(checkpoints_dir)+'/'+str(epoch), exist_ok=True)
            savepath = str(checkpoints_dir) +'/'+str(epoch)+ '/best_model.pth'
            log_string(logger,'Saving at %s' % savepath)
            state = {
                'epoch': epoch+1,
                'best err_axis': err_l2,
                'best err_rad': err_rad,
                'best accu':accu,
                'model_state_dict': model.state_dict()
            }
            torch.save(state, savepath)

            savepath = str(checkpoints_dir) + '/best_model.pth'
            torch.save(state, savepath)

            best_out_para_r = temp_out_para_r
            best_out_point = temp_out_point

    # os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    setup_seed(666)

    if args.data_type == 'complete':
        DATA_PATH = "%s/shapenet_new/" % (args.data_dir)
        CATES = ['02691156', '02933112', '02958343', '03001627','03636649', '04256520', '04379243', '04530566']
    elif args.data_type == 'partial':
        DATA_PATH = "%s/shapenet_partial_new/" % (args.data_dir)
        CATES = ['02691156', '02933112', '02958343', '03001627','03636649', '04256520', '04379243', '04530566']
    elif args.data_type == 'single_scan':
        DATA_PATH = "%s/shapenet_single_scan/" % (args.data_dir)
        CATES = ['02691156', '02933112', '02958343', '03001627','03636649', '04256520', '04379243', '04530566']
        # CATES = ['03001627', ]
        # CATES = ['04256520', ]

    elif args.data_type == 'uprl':
        DATA_PATH = "%s/upright16_data/" % (args.data_dir)
        CATES = ['airplane','bathtub', 'bicycle', 'car', 'chair', 'cup', 'dog', 'fruit','person','table']
    elif args.data_type == 'tta':
        DATA_PATH = "%s/upright16_data/testTTA/" % (args.data_dir)
        CATES = ['airplane','bathtub', 'bicycle', 'car', 'chair', 'cup', 'dog', 'fruit','person','table']

    
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))

    if args.note == 'debug':
        save_folder =  args.note
    else:
        save_folder =  str(args.rot_angle) + '_' + args.data_type + '_'  + args.loss + '_' + args.note + '_' + timestr

    experiment_dir = os.path.join(args.result_dir, args.task, save_folder)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_dir = os.path.join( experiment_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    log_dir = os.path.join( experiment_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    plot_dir = os.path.join( experiment_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    if args.use_tensorboard:
        summary_writer = SummaryWriter(log_dir)
    
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    '''DATA LOADING'''
    if 'single_scan' in DATA_PATH:
        DataLoader = DataLoader_Transform_Single_Scan
    elif 'TTA' in DATA_PATH:
        DataLoader = DataLoader_Transform_TTA
    elif 'upright' in DATA_PATH:
        DataLoader = DataLoader_Transform_Upright
    else:
        DataLoader = DataLoader_Transform
    
    if args.task == 'train':
        TRAIN_DATASET = DataLoader(root=DATA_PATH, npoint=args.num_point, split='train', category=CATES)
        TEST_DATASET = DataLoader(root=DATA_PATH, npoint=args.num_point, split='val', category=CATES)
        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    else:
        TEST_DATASET = DataLoader(root=DATA_PATH, npoint=args.num_point, split='test', category=CATES)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(logger,'PARAMETER ...')
    log_string(logger,args)

    '''MODEL LOADING'''
    
    model = ActorCritic(args.action_num).cuda()

    # model = nn.DataParallel(model)

    if args.task == 'train':
        log_string(logger,'length of TRAIN_DATASET:%d' % len(TRAIN_DATASET))

    log_string(logger,'length of TEST_DATASET:%d' % len(TEST_DATASET))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    log_string(logger,optimizer)
    log_string(logger,scheduler)

    global_epoch = 0
    start_epoch = 0
    
    if args.pretrain_dir is not None:
    # try:
        checkpoint = torch.load( os.path.join( args.pretrain_dir, "best_model.pth"))

        # checkpoint_state={}
        # for i in checkpoint['model_state_dict']:
        #     if'module.' in i:
        #         checkpoint_state[i[7:]] = checkpoint['model_state_dict'][i]
        # model.load_state_dict(checkpoint_state)
        
        model.load_state_dict(checkpoint['model_state_dict'])

        start_epoch=checkpoint['epoch']
        
        log_string(logger,'Use pretrain model %s' % args.pretrain_dir)
    # except:
    #     log_string(logger,'WARNING: No existing model, starting training from scratch...')
    #     start_epoch=0

    ave_train_policy_loss_all = []
    ave_train_value_loss_all = []
    ave_train_total_rewards_all = []
    loss_total_all = []

    ave_err_axis_all = []
    ave_err_angle_all = []

    ave_train_err_axis_all = []
    ave_train_err_angle_all = []

    acc_15_all = []
    acc_10_all = []
    acc_5_all = []

    '''TRAINING'''

    if args.task == 'train':
        logger.info('Start training...')

        import time
        
        for epoch in range(start_epoch, args.n_epochs):

            start = time.time()

            log_string(logger,'\nEpoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.n_epochs))
            scheduler.step()
            
            train(args, 'train', trainDataLoader, model, optimizer, epoch, summary_writer)
            end = time.time()
            print( "--------- %.3fs ---------" % (end-start) )

            test(args, 'val', testDataLoader, model, epoch)
            global_epoch += 1


    elif args.task == 'test':
        test(args, 'test', testDataLoader, model, 0)

    if args.task == 'test':
        save_file = "test.h5"
    else:
        save_file = "val.h5"

    # save the test/val results
    with h5py.File(os.path.join(experiment_dir, save_file), 'w') as f:
        f['gt_pts'] = best_origin_point
        f['gt_up'] = best_gt_para_r
        f['out_pts'] = best_out_point
        f['out_up'] = best_out_para_r


if __name__ == '__main__':

    args = parse_args()
    main(args)
