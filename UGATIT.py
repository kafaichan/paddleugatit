import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from custom_op import L1Loss, BCEWithLogitsLoss, LinearLrCoolDown

import time, itertools
from dataset import MyDatasetReader
from networks import *
from utils import *
from glob import glob

class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light
        self.args = args

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        self.trainA = MyDatasetReader(os.path.join('dataset', self.dataset, 'trainA'), self.args)
        self.trainB = MyDatasetReader(os.path.join('dataset', self.dataset, 'trainB'), self.args)
        self.testA = MyDatasetReader(os.path.join('dataset', self.dataset, 'testA'), self.args)
        self.testB = MyDatasetReader(os.path.join('dataset', self.dataset, 'testB'), self.args)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)

        """ Define Loss """
        self.L1_loss = L1Loss()
        self.MSE_loss = fluid.dygraph.MSELoss()
        self.BCE_loss = BCEWithLogitsLoss()

        """ Trainer """
        self.start_iter = 1
        if self.resume:
            print(os.path.join(self.result_dir, self.dataset, 'model', '*.pdparams'))
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pdparams'))
            if not len(model_list) == 0:
                model_list.sort()
                self.start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), self.start_iter)
                print(" [*] Load SUCCESS")

        if self.args.decay_flag:
            self.G_optim = fluid.optimizer.AdamOptimizer(learning_rate=LinearLrCoolDown(self.args.lr, self.args.iteration // 2, begin=self.start_iter), 
                parameter_list=list(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters())), 
                beta1=0.5, beta2=0.999, regularization=fluid.regularizer.L2Decay(self.weight_decay))

            self.D_optim = fluid.optimizer.AdamOptimizer(learning_rate=LinearLrCoolDown(self.args.lr, self.args.iteration // 2, begin=self.start_iter),
                parameter_list=list(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters())), 
                beta1=0.5, beta2=0.999, regularization=fluid.regularizer.L2Decay(self.weight_decay))
        else:
            self.G_optim = fluid.optimizer.AdamOptimizer(learning_rate=self.args.lr,
                parameter_list=list(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters())), 
                beta1=0.5, beta2=0.999, regularization=fluid.regularizer.L2Decay(self.weight_decay))

            self.D_optim = fluid.optimizer.AdamOptimizer(learning_rate=self.lr,
                parameter_list=list(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters())), 
                beta1=0.5, beta2=0.999, regularization=fluid.regularizer.L2Decay(self.weight_decay))


    def train(self):
        place = fluid.CUDAPlace(0) if self.device == 'cuda' else fluid.CPUPlace()

        with fluid.dygraph.guard(place):
            self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            # training loop
            print('training start !')
            start_time = time.time()
            for step in range(self.start_iter, self.iteration + 1):
                real_A = self.trainA.get_batch()
                real_B = self.trainB.get_batch()

                real_A, real_B = to_variable(real_A), to_variable(real_B)

                self.D_optim.clear_gradients()
                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)

                real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                D_ad_loss_GA = self.MSE_loss(real_GA_logit, fluid.layers.ones_like(real_GA_logit)) + self.MSE_loss(fake_GA_logit, fluid.layers.zeros_like(fake_GA_logit))
                D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, fluid.layers.ones_like(real_GA_cam_logit)) + self.MSE_loss(fake_GA_cam_logit, fluid.layers.zeros_like(fake_GA_cam_logit))
                D_ad_loss_LA = self.MSE_loss(real_LA_logit, fluid.layers.ones_like(real_LA_logit)) + self.MSE_loss(fake_LA_logit, fluid.layers.zeros_like(fake_LA_logit))
                D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, fluid.layers.ones_like(real_LA_cam_logit)) + self.MSE_loss(fake_LA_cam_logit, fluid.layers.zeros_like(fake_LA_cam_logit))
                D_ad_loss_GB = self.MSE_loss(real_GB_logit, fluid.layers.ones_like(real_GB_logit)) + self.MSE_loss(fake_GB_logit, fluid.layers.zeros_like(fake_GB_logit))
                D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, fluid.layers.ones_like(real_GB_cam_logit)) + self.MSE_loss(fake_GB_cam_logit, fluid.layers.zeros_like(fake_GB_cam_logit))
                D_ad_loss_LB = self.MSE_loss(real_LB_logit, fluid.layers.ones_like(real_LB_logit)) + self.MSE_loss(fake_LB_logit, fluid.layers.zeros_like(fake_LB_logit))
                D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, fluid.layers.ones_like(real_LB_cam_logit)) + self.MSE_loss(fake_LB_cam_logit, fluid.layers.zeros_like(fake_LB_cam_logit))

                D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
                D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

                Discriminator_loss = D_loss_A + D_loss_B

                # Update D
                Discriminator_loss.backward()
                self.D_optim.minimize(Discriminator_loss)

                # Update G
                self.G_optim.clear_gradients()

                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)

                fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)            
                fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                G_ad_loss_GA = self.MSE_loss(fake_GA_logit, fluid.layers.ones_like(fake_GA_logit))
                G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, fluid.layers.ones_like(fake_GA_cam_logit))
                G_ad_loss_LA = self.MSE_loss(fake_LA_logit, fluid.layers.ones_like(fake_LA_logit))
                G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, fluid.layers.ones_like(fake_LA_cam_logit))
                G_ad_loss_GB = self.MSE_loss(fake_GB_logit, fluid.layers.ones_like(fake_GB_logit))
                G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, fluid.layers.ones_like(fake_GB_cam_logit))
                G_ad_loss_LB = self.MSE_loss(fake_LB_logit, fluid.layers.ones_like(fake_LB_logit))
                G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, fluid.layers.ones_like(fake_LB_cam_logit))

                G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

                G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

                G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, fluid.layers.ones_like(fake_B2A_cam_logit)) + self.BCE_loss(fake_A2A_cam_logit, fluid.layers.zeros_like(fake_A2A_cam_logit))
                G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, fluid.layers.ones_like(fake_A2B_cam_logit)) + self.BCE_loss(fake_B2B_cam_logit, fluid.layers.zeros_like(fake_B2B_cam_logit))
                G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
                G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

                Generator_loss = G_loss_A + G_loss_B
                Generator_loss.backward()
                self.G_optim.minimize(Generator_loss)

                self.genA2B.clip_rho(0, 1.0)
                self.genB2A.clip_rho(0, 1.0)

                print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss.numpy(), Generator_loss.numpy()))
                if step % self.print_freq == 0:
                    train_sample_num = 5
                    test_sample_num = 5
                    A2B = np.zeros((self.img_size * 7, 0, 3))
                    B2A = np.zeros((self.img_size * 7, 0, 3))

                    self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                    for _ in range(train_sample_num):
                        real_A = self.trainA.get_batch()
                        real_B = self.trainB.get_batch()

                        real_A, real_B = to_variable(real_A), to_variable(real_B)

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A.numpy()[0]))),
                                                                   cam(tensor2numpy(fake_A2A_heatmap.numpy()[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A.numpy()[0]))),
                                                                   cam(tensor2numpy(fake_A2B_heatmap.numpy()[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B.numpy()[0]))),
                                                                   cam(tensor2numpy(fake_A2B2A_heatmap.numpy()[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A.numpy()[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B.numpy()[0]))),
                                                                   cam(tensor2numpy(fake_B2B_heatmap.numpy()[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B.numpy()[0]))),
                                                                   cam(tensor2numpy(fake_B2A_heatmap.numpy()[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A.numpy()[0]))),
                                                                   cam(tensor2numpy(fake_B2A2B_heatmap.numpy()[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B.numpy()[0])))), 0)), 1)

                    for _ in range(test_sample_num):
                        real_A = self.testA.get_batch()
                        real_B = self.testB.get_batch()

                        real_A, real_B = to_variable(real_A), to_variable(real_B)

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A.numpy()[0]))),
                                                                   cam(tensor2numpy(fake_A2A_heatmap.numpy()[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A.numpy()[0]))),
                                                                   cam(tensor2numpy(fake_A2B_heatmap.numpy()[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B.numpy()[0]))),
                                                                   cam(tensor2numpy(fake_A2B2A_heatmap.numpy()[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A.numpy()[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B.numpy()[0]))),
                                                                   cam(tensor2numpy(fake_B2B_heatmap.numpy()[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B.numpy()[0]))),
                                                                   cam(tensor2numpy(fake_B2A_heatmap.numpy()[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A.numpy()[0]))),
                                                                   cam(tensor2numpy(fake_B2A2B_heatmap.numpy()[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B.numpy()[0])))), 0)), 1)

                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                    self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

                if step % self.save_freq == 0:
                    self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

                if step % 1000 == 0:
                    fluid.dygraph.save_dygraph(self.genA2B.state_dict(), os.path.join(self.result_dir, self.dataset, 'gena2b_params_latest'))
                    fluid.dygraph.save_dygraph(self.genB2A.state_dict(), os.path.join(self.result_dir, self.dataset, 'genb2a_params_latest'))
                    fluid.dygraph.save_dygraph(self.disGA.state_dict(), os.path.join(self.result_dir, self.dataset, 'disga_params_latest'))
                    fluid.dygraph.save_dygraph(self.disGB.state_dict(), os.path.join(self.result_dir, self.dataset, 'disgb_params_latest'))
                    fluid.dygraph.save_dygraph(self.disLA.state_dict(), os.path.join(self.result_dir, self.dataset, 'disla_params_latest'))
                    fluid.dygraph.save_dygraph(self.disLB.state_dict(), os.path.join(self.result_dir, self.dataset, 'dislb_params_latest'))
                    

    def save(self, dir, step):
        fluid.dygraph.save_dygraph(self.genA2B.state_dict(), os.path.join(dir, 'gena2b_params_%07d' % step))
        fluid.dygraph.save_dygraph(self.genB2A.state_dict(), os.path.join(dir, 'genb2a_params_%07d' % step))
        fluid.dygraph.save_dygraph(self.disGA.state_dict(), os.path.join(dir, 'disga_params_%07d' % step))
        fluid.dygraph.save_dygraph(self.disGB.state_dict(), os.path.join(dir, 'disgb_params_%07d' % step))
        fluid.dygraph.save_dygraph(self.disLA.state_dict(), os.path.join(dir, 'disla_params_%07d' % step))
        fluid.dygraph.save_dygraph(self.disLB.state_dict(), os.path.join(dir, 'dislb_params_%07d' % step))

    def load(self, dir, step, test=False):
        genA2B_param, _ = fluid.dygraph.load_dygraph(os.path.join(dir, 'gena2b_params_%07d' % step))
        genB2A_param, _ = fluid.dygraph.load_dygraph(os.path.join(dir, 'genb2a_params_%07d' % step))
        self.genA2B.set_dict(genA2B_param)
        self.genB2A.set_dict(genB2A_param)

        if not test:
            disGA_param, _ = fluid.dygraph.load_dygraph(os.path.join(dir, 'disga_params_%07d' % step))
            disGB_param, _ = fluid.dygraph.load_dygraph(os.path.join(dir, 'disgb_params_%07d' % step))
            disLA_param, _ = fluid.dygraph.load_dygraph(os.path.join(dir, 'disla_params_%07d' % step))
            disLB_param, _ = fluid.dygraph.load_dygraph(os.path.join(dir, 'dislb_params_%07d' % step))
            self.disGA.set_dict(disGA_param)
            self.disGB.set_dict(disGB_param)
            self.disLA.set_dict(disLA_param)
            self.disLB.set_dict(disLB_param)


    def test(self):
        place = fluid.CUDAPlace(0) if self.device == 'cuda' else fluid.CPUPlace()

        with fluid.dygraph.guard(place):            
            self.genA2B.eval(), self.genB2A.eval()
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pdparams'))
            if not len(model_list) == 0:
                model_list.sort()
                iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter, test=True)
                print(" [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return

            batch_idx = 1
            while True:
                real_A = self.testA.get_batch(False)
                real_A = to_variable(real_A)                
                fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

                A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B.numpy()[0])))
                A2B2A = RGB2BGR(tensor2numpy(denorm(fake_A2B2A.numpy()[0])))
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%03d.png' % (batch_idx)), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B2A_%03d.png' % (batch_idx)), A2B2A * 255.0)
                batch_idx += 1
                if self.testA.idx == 0: break


            batch_idx = 1
            while True:
                real_B = self.testB.get_batch(False)
                real_B = to_variable(real_B)
                fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)
                fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A.numpy()[0])))
                B2A2B = RGB2BGR(tensor2numpy(denorm(fake_B2A2B.numpy()[0])))
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%03d.png' % (batch_idx)), B2A * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A2B_%03d.png' % (batch_idx)), B2A2B * 255.0)

                batch_idx += 1
                if self.testB.idx == 0: break