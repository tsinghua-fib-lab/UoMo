import torch
from torch.optim import AdamW
import random
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import time


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class TrainLoop:
    def __init__(self, args, writer, model, diffusion, data, test_data, val_data, device):
        self.args = args
        self.writer = writer
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.test_data = test_data
        self.val_data = val_data
        self.device = device
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.opt = AdamW([p for p in self.model.parameters() if p.requires_grad==True], lr=args.lr, weight_decay=self.weight_decay)
        self.log_interval = args.log_interval
        self.best_rmse_random = 1e9
        self.warmup_steps=5
        self.min_lr = args.min_lr
        self.best_rmse = 1e9
        self.early_stop = 0
        
        self.mask_list = {'random_masking':[0.5],'generation_masking':[0.25],'short_long_temporal_masking':[0.25,0.75]}


    def run_step(self, batch, step, index, mask_stg, mask_rate, name):
        self.opt.zero_grad()
        loss, num = self.forward_backward(batch, step, index=index, mask_stg = mask_stg, mask_rate = mask_rate, name = name)

        self._anneal_lr()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
        self.opt.step()
        return loss, num

    def Sample(self, test_data, step, mask_stg, mask_rate, seed=None, dataset='', index=0, Type='val'):
        
        with torch.no_grad():
            error_mae, error_norm, error, num = 0.0, 0.0, 0.0, 0.0

            for _, batch in enumerate(test_data[index]):
                
                loss= self.model_forward(batch, self.model, mask_stg, mask_rate, seed=seed, data = dataset, mode='forward')
                # error_norm += loss.item()
                error_norm += sum(loss['loss'])
                #
                num += loss['loss'].shape[0]
                # num2 += (1-mask).sum().item()


        loss_test = error_norm / num

        return loss_test


    def Evaluation(self, test_data, epoch, seed=None, best=True, Type='val'):


        rmse_list = []
        rmse_key_result = {}

        for index, dataset_name in enumerate(self.args.dataset.split('*')):

            rmse_key_result[dataset_name] = {}


            for s in self.mask_list:
                for m in self.mask_list[s]:
                    result= self.Sample(test_data, epoch, mask_stg = s, mask_rate = m,seed=seed, dataset = dataset_name, index=index, Type=Type)
                    rmse_list.append(result)
                    if s not in rmse_key_result[dataset_name]:
                        rmse_key_result[dataset_name][s] = {}
                    rmse_key_result[dataset_name][s][m] = result
                    if Type == 'val':
                        self.writer.add_scalar('Evaluation/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), result, epoch)
                    elif Type == 'test':
                        self.writer.add_scalar('Test_RMSE/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), result, epoch)


        loss_test = np.mean(np.array([tensor.cpu().numpy() for tensor in rmse_list]))

        if best:
            is_break = self.best_model_save(epoch, loss_test, rmse_key_result)
            return is_break

        else:
            return  loss_test, rmse_key_result

    def best_model_save(self, step, rmse, rmse_key_result):
        if rmse < self.best_rmse:
            self.early_stop = 0
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best_stage_{}.pkl'.format(self.args.stage))
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best.pkl')
            self.best_rmse = rmse
            self.writer.add_scalar('Evaluation/RMSE_best', self.best_rmse, step)
            print('\nRMSE_best:{}\n'.format(self.best_rmse))
            print(str(rmse_key_result)+'\n')
            with open(self.args.model_path+'result.txt', 'w') as f:
                f.write('stage:{}, epoch:{}, best rmse: {}\n'.format(self.args.stage, step, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('stage:{}, epoch:{}, best rmse: {}\n'.format(self.args.stage, step, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            return 'save'

        elif self.args.stage in [0,1,2]:
            self.early_stop += 1
            print('\nRMSE:{}, RMSE_best:{}, early_stop:{}\n'.format(rmse, self.best_rmse, self.early_stop))
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('RMSE:{}, not optimized, early_stop:{}\n'.format(rmse, self.early_stop))
            if self.early_stop >= self.args.early_stop:
                print('Early stop!')
                print('Generate samples:')
                self.model.eval()
                model_path = self.args.model_path + 'model_save/model_best.pkl'
                self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
                print('Load model success')

                error_before_scaler, error_after_scaler = 0.0, 0.0
                target_to_save = []
                sample_to_save = []
                before_target_to_save = []
                before_sample_to_save = []
                mask_to_save = []

                for index_t, dataset_name_t in enumerate(self.args.dataset.split('*')):
                    for index_mask, batch2 in enumerate(self.test_data[index_t]):
                        model_kwargs_t = dict(y=batch2[1].to(device=self.device))
                        x_start = batch2[0].to(device=self.device)

                        if index_mask % 3 == 0:
                            mask_strategy = 'short_long_temporal_masking'
                            mask_ratio = 0.75

                        elif index_mask % 3 == 1:
                            mask_strategy = 'short_long_temporal_masking'
                            mask_ratio = 0.25
                        else:
                            mask_strategy = 'generation_masking'
                            mask_ratio = 0.25

                        mask_origin = self.function_dict[mask_strategy](self, x_start, mask_ratio=mask_ratio)
                        x_start_masked = mask_origin * x_start

                        sample, mask = self.diffusion.p_sample_loop(
                            self.model, batch2[0].shape, x_start, batch2[2], batch2[3], mask_origin, x_start_masked,
                            clip_denoised=True, model_kwargs=model_kwargs_t, progress=True,
                            device=self.device
                        )
                        target = batch2[0]
                        shape_base = target.shape
                        samples = sample * mask.to(device=self.device) + target.to(device=self.device) * (1 - mask)

                        before_target_to_save.append(target.detach().cpu().numpy())
                        before_sample_to_save.append(samples.detach().cpu().numpy())

                        error_before_scaler += mean_squared_error(samples.reshape(-1, 1).detach().cpu().numpy(),
                                                                  target.reshape(-1, 1).detach().cpu().numpy(),
                                                                  squared=False) / batch2[0].shape[0]
                        error_after_scaler += mean_squared_error(self.args.scaler[dataset_name_t].inverse_transform(
                            samples.reshape(-1, 1).detach().cpu().numpy()),
                                                                 self.args.scaler[dataset_name_t].inverse_transform(
                                                                     target.reshape(-1, 1).detach().cpu().numpy()),
                                                                 squared=False) / batch2[0].shape[0]

                        save_tar = self.args.scaler[dataset_name_t].inverse_transform(
                            target.reshape(-1, 1).detach().cpu().numpy()).reshape(shape_base).squeeze(1)
                        save_gen = self.args.scaler[dataset_name_t].inverse_transform(
                            samples.reshape(-1, 1).detach().cpu().numpy()).reshape(shape_base).squeeze(1)
                        target_to_save.append(save_tar)
                        sample_to_save.append(save_gen)
                        mask_to_save.append(mask_origin.squeeze(1).detach().cpu().numpy())
                print('error_before_scaler:', error_before_scaler)
                print('error_after_scaler:', error_after_scaler)

                filename_gen = self.args.model_path + 'generate_{}_{}.npz'.format(self.args.dataset.replace('*', '_'),
                                                                                  self.args.length0)
                filename_tar = self.args.model_path + 'target_{}_{}.npz'.format(self.args.dataset.replace('*', '_'),
                                                                                self.args.length0)
                filename_mask = self.args.model_path + 'mask_{}_{}.npz'.format(self.args.dataset.replace('*', '_'),
                                                                               self.args.length0)

                np.savez(filename_gen, gen_traffic=sample_to_save)
                np.savez(filename_tar, tar_traffic=target_to_save)
                np.savez(filename_mask, mask=mask_to_save)
                with open(self.args.model_path+'result.txt', 'a') as f:
                    f.write('Early stop!\n')
                with open(self.args.model_path+'result_all.txt', 'a') as f:
                    f.write('Early stop!\n')
                exit()
            return 'none'
        
    def mask_select(self):

            
        mask_strategy=random.choice(['random_masking','generation_masking','short_long_temporal_masking'])
        mask_ratio=random.choice(self.mask_list[mask_strategy])

        return mask_strategy, mask_ratio

    def run_loop(self):
        step = 0
        
        self.Evaluation(self.val_data, 0, best=True, Type='val')
        for epoch in range(self.args.total_epoches):
            print('Training')

            self.step = epoch
            
            loss_all, num_all = 0.0, 0.0
            start = time.time()
            for name, batch in self.data:
                mask_strategy, mask_ratio = self.mask_select()
                loss, num = self.run_step(batch, step,index=0, mask_stg =mask_strategy, mask_rate =  mask_ratio, name = name)
                step += 1
                loss_all += loss * num
                num_all += num

            end = time.time()
            print('training time:{} min'.format(round((end-start)/60.0,2)))
            print('epoch:{}, training loss:{}'.format(epoch, loss_all / num_all))

            if epoch >= 10 :
                self.writer.add_scalar('Training/Stage_{}_Loss_epoch'.format(self.args.stage), loss_all / num_all, epoch)


            if epoch % self.log_interval == 0 and epoch > 0 or epoch == 10 or epoch == self.args.total_epoches-1:
                print('Evaluation')
                is_break = self.Evaluation(self.val_data, epoch, best=True, Type='val')

                if is_break == 'break_1_stage':
                    break

                if is_break == 'save':
                    print('test evaluate!')
                    #rmse_test, rmse_key_test = self.Evaluation(self.test_data, epoch, best=False, Type='test')
 
                    rmse_test, rmse_key_test = self.Evaluation(self.test_data, epoch, best=False, Type='test')
                    print('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                    print(str(rmse_key_test)+'\n')
                    with open(self.args.model_path+'result.txt', 'a') as f:
                        f.write('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                        f.write(str(rmse_key_test)+'\n')
                    with open(self.args.model_path+'result_all.txt', 'a') as f:
                        f.write('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                        f.write(str(rmse_key_test)+'\n')
        print('Generate samples:')
        self.model.eval()
        model_path = self.args.model_path+'model_save/model_best.pkl'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
        print('Load model success')
        
        error_before_scaler, error_after_scaler =0.0, 0.0
        target_to_save = []
        sample_to_save = []
        before_target_to_save = []
        before_sample_to_save = []
        mask_to_save = []
        
        for index_t, dataset_name_t in enumerate(self.args.dataset.split('*')):
            for index_mask, batch2 in enumerate(self.test_data[index_t]):
                model_kwargs_t = dict(y=batch2[1].to(device=self.device))
                x_start = batch2[0].to(device=self.device)

                if index_mask % 3 ==0 :
                    mask_strategy = 'short_long_temporal_masking'
                    mask_ratio = 0.75

                elif index_mask % 3 ==1 :
                    mask_strategy = 'short_long_temporal_masking'
                    mask_ratio = 0.25
                else:
                    mask_strategy = 'generation_masking'
                    mask_ratio = 0.25                   

                mask_origin = self.function_dict[mask_strategy](self, x_start, mask_ratio=mask_ratio)
                x_start_masked = mask_origin * x_start


                sample, mask = self.diffusion.p_sample_loop(
                    self.model, batch2[0].shape, x_start, batch2[2],batch2[3], mask_origin,x_start_masked, clip_denoised=True, model_kwargs=model_kwargs_t, progress=True,
                    device=self.device
                )
                target = batch2[0]
                shape_base = target.shape
                samples = sample * mask.to(device=self.device) + target.to(device=self.device) * (1-mask)

                before_target_to_save.append(target.detach().cpu().numpy())
                before_sample_to_save.append(samples.detach().cpu().numpy())

                error_before_scaler += mean_squared_error(samples.reshape(-1,1).detach().cpu().numpy(),target.reshape(-1,1).detach().cpu().numpy(), squared=False)/batch2[0].shape[0]
                error_after_scaler += mean_squared_error(self.args.scaler[dataset_name_t].inverse_transform(samples.reshape(-1,1).detach().cpu().numpy()),self.args.scaler[dataset_name_t].inverse_transform(target.reshape(-1,1).detach().cpu().numpy()), squared=False)/batch2[0].shape[0]


                save_tar = self.args.scaler[dataset_name_t].inverse_transform(target.reshape(-1,1).detach().cpu().numpy()).reshape(shape_base).squeeze(1)
                save_gen = self.args.scaler[dataset_name_t].inverse_transform(samples.reshape(-1,1).detach().cpu().numpy()).reshape(shape_base).squeeze(1)
                target_to_save.append(save_tar)
                sample_to_save.append(save_gen)
                mask_to_save.append(mask_origin.squeeze(1).detach().cpu().numpy())
        print('error_before_scaler:', error_before_scaler)
        print('error_after_scaler:', error_after_scaler)


        filename_gen = self.args.model_path + 'generate_{}_{}.npz'.format(self.args.dataset.replace('*', '_'), self.args.length0)
        filename_tar = self.args.model_path+ 'target_{}_{}.npz'.format(self.args.dataset.replace('*', '_'), self.args.length0)
        filename_mask = self.args.model_path+ 'mask_{}_{}.npz'.format(self.args.dataset.replace('*', '_'), self.args.length0)

        np.savez(filename_gen, gen_traffic=sample_to_save)
        np.savez(filename_tar, tar_traffic=target_to_save)
        np.savez(filename_mask, mask=mask_to_save)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        B, _, T, H, W = x.shape  # batch, length,
        x = x.reshape(B, -1, self.args.t_patch_size * self.args.patch_size ** 2)
        B, C, _ = x.shape
        num_elements = C
        # num_elements = L * H * W
        num_ones = int(num_elements * mask_ratio)

        mask = torch.zeros_like(x.squeeze(1), dtype=torch.bool)
        # torch.manual_seed(123)
        for b in range(B):
            # Create a flattened array of indices and shuffle it
            indices = torch.randperm(num_elements, device=x.device)
            # Set the first num_ones indices to 1
            ones_indices = indices[:num_ones]
            for j in ones_indices:
                mask[b, j, :] = 1
        mask = mask.reshape(B, 1, T, H, W)
        return mask.float()

    def small_tube_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        B, _, T, H, W = x.shape
        t = T // self.args.t_patch_size
        h = H // self.args.patch_size
        w = W // self.args.patch_size
        x = x.reshape(B, 1, t, h, w, self.args.t_patch_size * self.args.patch_size ** 2)

        mask = torch.zeros_like(x, dtype=torch.float32, device=x.device)

        num_to_mask = int(h * w * mask_ratio)  # 计算需要mask的位置数量

        for b in range(B):
            # 为每个batch随机选择位置
            mask_indices = torch.randperm(h * w, device=x.device)[:num_to_mask]
            for idx in mask_indices:
                hs = idx // w  # 利用总列数W计算行索引
                ws = idx % w  # 计算列索引
                mask[b, :, :, hs, ws, :] = 1  # 设置选中位置的所有时间为1
        mask = mask.reshape(B, 1, T, H, W)
        return mask.float()

    def short_long_temporal_masking(self, x, mask_ratio):
        """
        根据 mask_ratio大小控制短时间mask和长时间mask
        """
        B, _, T, H, W = x.shape
        t = T // self.args.t_patch_size
        h = H // self.args.patch_size
        w = W // self.args.patch_size
        x = x.reshape(B, 1, t, h, w, self.args.t_patch_size * self.args.patch_size ** 2)

        mask = torch.zeros_like(x, dtype=torch.float32, device=x.device)

        num_times_to_mask = int(t * mask_ratio)  # 计算需要mask的时间步数
        start_time = t - num_times_to_mask  # 计算时间维度的mask开始点

        # 为所有空间位置的最后T*m个时间步设置mask
        mask[:, :, start_time:, :, :, :] = 1
        mask = mask.reshape(B, 1, T, H, W)
        return mask.float()

    function_dict = {
        'random_masking': random_masking,
        'generation_masking': small_tube_masking,
        'short_long_temporal_masking': short_long_temporal_masking
    }

    def model_forward(self, batch, model, mask_stg, mask_rate, seed = None, data=None, mode='backward'):
        self.args.name_id = data
        batch = [i.to(self.device) for i in batch]
        t = torch.randint(0, self.diffusion.num_timesteps, (batch[0].shape[0],), device=self.device)
        x_start = batch[0]
        model_kwargs = dict(y=batch[1])

        mask_origin = self.function_dict[mask_stg](self,x_start, mask_ratio = mask_rate)
        loss= self.diffusion.training_losses(model, x_start, batch[2], batch[3], mask_origin, t, model_kwargs)

        return loss


    def forward_backward(self, batch, step, index, mask_stg, mask_rate, name=None):

        loss_multi= self.model_forward(batch, self.model, mask_stg = mask_stg, mask_rate = mask_rate, data=name, mode='backward')

        num = loss_multi['loss'].shape[0]
        loss = sum(loss_multi['loss'])/num
        loss.backward()

        self.writer.add_scalar('Training/Loss_step', np.sqrt(loss.detach().cpu().numpy()), step)
        return loss.item(), num

    def _anneal_lr(self):
        if self.step < self.warmup_steps:
            lr = self.lr * (self.step+1) / self.warmup_steps
        elif self.step < self.lr_anneal_steps:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (self.step - self.warmup_steps)
                    / (self.lr_anneal_steps - self.warmup_steps)
                )
            )
        else:
            lr = self.min_lr
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
        self.writer.add_scalar('Training/LR', lr, self.step)
        return lr

