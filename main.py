from utils import *
import time
import paddle
import paddle.fluid as fluid
from networks import *
from utils import *
from glob import glob
from dataset import MyDatasetReader
import os


def get_params_by_prefix(program, prefix):
    all_params = program.global_block().all_parameters()
    return [t.name for t in all_params if t.name.startswith(prefix)]

def get_params_by_suffix(program, suffix):
    all_params = program.global_block().all_parameters()
    return [t.name for t in all_params if t.name.endswith(suffix)]

################################################################################################################################################################################################
args = parse_args()

startup_program = fluid.default_startup_program()
generator_program = fluid.Program()
discriminator_program = fluid.Program()

# Discriminator
with fluid.program_guard(discriminator_program, startup_program):
    data_shape = [None, 3, args.img_size, args.img_size]
    real_A = fluid.data(name='real_A', shape=data_shape, dtype='float32')
    real_B = fluid.data(name='real_B', shape=data_shape, dtype='float32')

    fake_A2B, _, _ = resnet_generator(name='GA2B', inputs=real_A, input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res, img_size=args.img_size, light=args.light)
    fake_B2A, _, _ = resnet_generator(name='GB2A', inputs=real_B, input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res, img_size=args.img_size, light=args.light)

    real_GA_logit, real_GA_cam_logit, _ = discriminator(name='DGA', inputs=real_A, input_nc=3, ndf=args.ch, n_layers=7)
    real_LA_logit, real_LA_cam_logit, _ = discriminator(name='DLA', inputs=real_A, input_nc=3, ndf=args.ch, n_layers=5)
    real_GB_logit, real_GB_cam_logit, _ = discriminator(name='DGB', inputs=real_B, input_nc=3, ndf=args.ch, n_layers=7)
    real_LB_logit, real_LB_cam_logit, _ = discriminator(name='DLB', inputs=real_B, input_nc=3, ndf=args.ch, n_layers=5)

    fake_GA_logit, fake_GA_cam_logit, _ = discriminator(name='DGA', inputs=fake_B2A, input_nc=3, ndf=args.ch, n_layers=7)
    fake_LA_logit, fake_LA_cam_logit, _ = discriminator(name='DLA', inputs=fake_B2A, input_nc=3, ndf=args.ch, n_layers=5)
    fake_GB_logit, fake_GB_cam_logit, _ = discriminator(name='DGB', inputs=fake_A2B, input_nc=3, ndf=args.ch, n_layers=7)
    fake_LB_logit, fake_LB_cam_logit, _ = discriminator(name='DLB', inputs=fake_A2B, input_nc=3, ndf=args.ch, n_layers=5)

    D_ad_loss_GA = fluid.layers.mse_loss(input=real_GA_logit, label=fluid.layers.ones_like(real_GA_logit)) + \
                   fluid.layers.mse_loss(input=fake_GA_logit, label=fluid.layers.zeros_like(fake_GA_logit))
    D_ad_cam_loss_GA = fluid.layers.mse_loss(input=real_GA_cam_logit, label=fluid.layers.ones_like(real_GA_cam_logit)) + \
                       fluid.layers.mse_loss(input=fake_GA_cam_logit, label=fluid.layers.zeros_like(fake_GA_cam_logit))
    D_ad_loss_LA = fluid.layers.mse_loss(input=real_LA_logit, label=fluid.layers.ones_like(real_LA_logit)) + \
                   fluid.layers.mse_loss(input=fake_LA_logit, label=fluid.layers.zeros_like(fake_LA_logit))
    D_ad_cam_loss_LA = fluid.layers.mse_loss(input=real_LA_cam_logit, label=fluid.layers.ones_like(real_LA_cam_logit)) + \
                       fluid.layers.mse_loss(input=fake_LA_cam_logit, label=fluid.layers.zeros_like(fake_LA_cam_logit))
    D_ad_loss_GB = fluid.layers.mse_loss(input=real_GB_logit, label=fluid.layers.ones_like(real_GB_logit)) + \
                   fluid.layers.mse_loss(input=fake_GB_logit, label=fluid.layers.zeros_like(fake_GB_logit))
    D_ad_cam_loss_GB = fluid.layers.mse_loss(input=real_GB_cam_logit, label=fluid.layers.ones_like(real_GB_cam_logit)) + \
                       fluid.layers.mse_loss(input=fake_GB_cam_logit, label=fluid.layers.zeros_like(fake_GB_cam_logit))
    D_ad_loss_LB = fluid.layers.mse_loss(input=real_LB_logit, label=fluid.layers.ones_like(real_LB_logit)) + \
                   fluid.layers.mse_loss(input=fake_LB_logit, label=fluid.layers.zeros_like(fake_LB_logit))
    D_ad_cam_loss_LB = fluid.layers.mse_loss(input=real_LB_cam_logit, label=fluid.layers.ones_like(real_LB_cam_logit)) + \
                       fluid.layers.mse_loss(input=fake_LB_cam_logit, label=fluid.layers.zeros_like(fake_LB_cam_logit))

    D_loss_A = args.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
    D_loss_B = args.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

    Discriminator_loss = D_loss_A + D_loss_B

    d_params = get_params_by_prefix(discriminator_program, "D")
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=args.lr, beta1=0.5, beta2=0.999, regularization=fluid.regularizer.L2Decay(args.weight_decay), name='dopt')
    optimizer.minimize(Discriminator_loss, parameter_list=d_params)


# Generator
with fluid.program_guard(generator_program, startup_program):
    data_shape = [None, 3, args.img_size, args.img_size]
    real_A = fluid.data(name='real_A', shape=data_shape, dtype='float32')
    real_B = fluid.data(name='real_B', shape=data_shape, dtype='float32')

    fake_A2B, fake_A2B_cam_logit, fake_A2B_heatmap = resnet_generator(name='GA2B', inputs=real_A, input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res, img_size=args.img_size, light=args.light)
    fake_B2A, fake_B2A_cam_logit, fake_B2A_heatmap = resnet_generator(name='GB2A', inputs=real_B, input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res, img_size=args.img_size, light=args.light)

    fake_A2B2A, _, fake_A2B2A_heatmap = resnet_generator(name='GB2A', inputs=fake_A2B, input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res, img_size=args.img_size, light=args.light)
    fake_B2A2B, _, fake_B2A2B_heatmap = resnet_generator(name='GA2B', inputs=fake_B2A, input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res, img_size=args.img_size, light=args.light)

    fake_A2A, fake_A2A_cam_logit, fake_A2A_heatmap = resnet_generator(name='GB2A', inputs=real_A, input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res, img_size=args.img_size, light=args.light)      
    fake_B2B, fake_B2B_cam_logit, fake_B2B_heatmap = resnet_generator(name='GA2B', inputs=real_B, input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res, img_size=args.img_size, light=args.light)

    test_program = generator_program.clone(for_test=True)

    fake_GA_logit, fake_GA_cam_logit, _ = discriminator(name='DGA', inputs=fake_B2A, input_nc=3, ndf=args.ch, n_layers=7)
    fake_LA_logit, fake_LA_cam_logit, _ = discriminator(name='DLA', inputs=fake_B2A, input_nc=3, ndf=args.ch, n_layers=5)
    fake_GB_logit, fake_GB_cam_logit, _ = discriminator(name='DGB', inputs=fake_A2B, input_nc=3, ndf=args.ch, n_layers=7)
    fake_LB_logit, fake_LB_cam_logit, _ = discriminator(name='DLB', inputs=fake_A2B, input_nc=3, ndf=args.ch, n_layers=5)

    G_ad_loss_GA = fluid.layers.mse_loss(input=fake_GA_logit, label=fluid.layers.ones_like(fake_GA_logit))
    G_ad_cam_loss_GA = fluid.layers.mse_loss(input=fake_GA_cam_logit, label=fluid.layers.ones_like(fake_GA_cam_logit))
    G_ad_loss_LA = fluid.layers.mse_loss(input=fake_LA_logit, label=fluid.layers.ones_like(fake_LA_logit))
    G_ad_cam_loss_LA = fluid.layers.mse_loss(input=fake_LA_cam_logit, label=fluid.layers.ones_like(fake_LA_cam_logit))
    G_ad_loss_GB = fluid.layers.mse_loss(input=fake_GB_logit, label=fluid.layers.ones_like(fake_GB_logit))
    
    G_ad_cam_loss_GB = fluid.layers.mse_loss(input=fake_GB_cam_logit, label=fluid.layers.ones_like(fake_GB_cam_logit))
    G_ad_loss_LB = fluid.layers.mse_loss(input=fake_LB_logit, label=fluid.layers.ones_like(fake_LB_logit))
    G_ad_cam_loss_LB = fluid.layers.mse_loss(input=fake_LB_cam_logit, label=fluid.layers.ones_like(fake_LB_cam_logit))

    G_recon_loss_A = l1loss(fake_A2B2A, real_A)
    G_recon_loss_B = l1loss(fake_B2A2B, real_B)
    G_identity_loss_A = l1loss(fake_A2A, real_A)
    G_identity_loss_B = l1loss(fake_B2B, real_B)

    G_cam_loss_A = bce_with_logit_loss(fake_B2A_cam_logit, fluid.layers.ones_like(fake_B2A_cam_logit)) + \
                   bce_with_logit_loss(fake_A2A_cam_logit, fluid.layers.zeros_like(fake_A2A_cam_logit))
    G_cam_loss_B = bce_with_logit_loss(fake_A2B_cam_logit, fluid.layers.ones_like(fake_A2B_cam_logit)) + \
                   bce_with_logit_loss(fake_B2B_cam_logit, fluid.layers.zeros_like(fake_B2B_cam_logit))
    G_loss_A =  args.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + args.cycle_weight * G_recon_loss_A + args.identity_weight * G_identity_loss_A + args.cam_weight * G_cam_loss_A
    G_loss_B = args.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + args.cycle_weight * G_recon_loss_B + args.identity_weight * G_identity_loss_B + args.cam_weight * G_cam_loss_B

    Generator_loss = G_loss_A + G_loss_B

    g_params = get_params_by_prefix(generator_program, "G")
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=args.lr, beta1=0.5, beta2=0.999, regularization=fluid.regularizer.L2Decay(args.weight_decay), name='gopt')
    optimizer.minimize(Generator_loss, parameter_list=g_params)


################################################################################################################################################################################################

place = fluid.CUDAPlace(0) if args.device == "cuda" else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_program)


trainA = MyDatasetReader(os.path.join('dataset', args.dataset, 'trainA'), args).create_reader()
trainB = MyDatasetReader(os.path.join('dataset', args.dataset, 'trainB'), args).create_reader()
testA = MyDatasetReader(os.path.join('dataset', args.dataset, 'testA'), args).create_reader()
testB = MyDatasetReader(os.path.join('dataset', args.dataset, 'testB'), args).create_reader()
trainA_loader = paddle.batch(paddle.reader.shuffle(trainA, 30000), batch_size=args.batch_size)
trainB_loader = paddle.batch(paddle.reader.shuffle(trainB, 30000), batch_size=args.batch_size)
testA_loader = paddle.batch(testA, batch_size=args.batch_size)
testB_loader = paddle.batch(testB, batch_size=args.batch_size)


if args.phase == 'train' :
    start_iter = 1

    print('training start !')
    start_time = time.time()

    for step in range(start_iter, args.iteration + 1):
        try:
            real_A, _ = zip(*next(trainA_loader()))
        except:
            trainA_iter = iter(trainA_loader())
            real_A, _ = zip(*next(trainA_iter))

        try:
            real_B, _ = zip(*next(trainB_loader()))
        except:
            trainB_iter = iter(trainB_loader())
            real_B, _ = zip(*next(trainB_iter))

        real_A, real_B = np.array(real_A), np.array(real_B)

        d_loss = exe.run(program=discriminator_program, fetch_list=[Discriminator_loss.name], feed={'real_A': real_A, 'real_B': real_B})
        g_loss = exe.run(program=generator_program, fetch_list=[Generator_loss.name], feed={'real_A': real_A, 'real_B': real_B})

        # Rho Clipping
        for param in get_params_by_suffix(generator_program, "_rho"):
            rho = fluid.global_scope().find_var(param).get_tensor()
            rho.set(np.clip(np.array(rho), 0, 1), place)

        print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, args.iteration, time.time() - start_time, d_loss[0], g_loss[0]))


        if step % args.print_freq == 0:
            A2B = np.zeros((args.img_size * 7, 0, 3))
            B2A = np.zeros((args.img_size * 7, 0, 3))

            for _ in range(5):
                try:
                    real_A, _ = zip(*next(trainA_loader()))
                except:
                    A_iter = iter(trainA_loader())
                    real_A, _ = zip(*next(A_iter))

                try:
                    real_B, _ = zip(*next(trainB_loader()))
                except:
                    B_iter = iter(trainB_loader())
                    real_B, _ = zip(*next(B_iter))

                real_A = np.array(real_A)
                real_B = np.array(real_B)

                gen_result = exe.run(program=test_program, feed={'real_A': real_A, 'real_B': real_B}, \
                    fetch_list=[fake_A2A_heatmap.name, fake_A2A.name, fake_A2B_heatmap.name, fake_A2B.name, fake_A2B2A_heatmap.name, \
                    fake_A2B2A.name, fake_B2B_heatmap.name, fake_B2B.name, fake_B2A_heatmap.name, fake_B2A.name,fake_B2A2B_heatmap.name, fake_B2A2B.name])

                A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                           cam(tensor2numpy(gen_result[0][0]), args.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(gen_result[1][0]))),
                                                           cam(tensor2numpy(gen_result[2][0]), args.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(gen_result[3][0]))),
                                                           cam(tensor2numpy(gen_result[4][0]), args.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(gen_result[5][0])))), 0)), 1)

                B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                           cam(tensor2numpy(gen_result[6][0]), args.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(gen_result[7][0]))),
                                                           cam(tensor2numpy(gen_result[8][0]), args.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(gen_result[9][0]))),
                                                           cam(tensor2numpy(gen_result[10][0]), args.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(gen_result[11][0])))), 0)), 1)

            for _ in range(5):
                try:
                    real_A, _ = zip(*next(testA_loader()))
                except:
                    A_iter = iter(testA_loader())
                    real_A, _ = zip(*next(A_iter))

                try:
                    real_B, _ = zip(*next(testB_loader()))
                except:
                    B_iter = iter(testB_loader())
                    real_B, _ = zip(*next(B_iter))

                real_A = np.array(real_A)
                real_B = np.array(real_B)

                gen_result = exe.run(program=test_program, feed={'real_A': real_A, 'real_B': real_B}, \
                    fetch_list=[fake_A2A_heatmap, fake_A2A, fake_A2B_heatmap, fake_A2B, fake_A2B2A_heatmap, fake_A2B2A, fake_B2B_heatmap, fake_B2B, fake_B2A_heatmap, fake_B2A,fake_B2A2B_heatmap, fake_B2A2B])

                A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                           cam(tensor2numpy(gen_result[0][0]), args.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(gen_result[1][0]))),
                                                           cam(tensor2numpy(gen_result[2][0]), args.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(gen_result[3][0]))),
                                                           cam(tensor2numpy(gen_result[4][0]), args.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(gen_result[5][0])))), 0)), 1)

                B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                           cam(tensor2numpy(gen_result[6][0]), args.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(gen_result[7][0]))),
                                                           cam(tensor2numpy(gen_result[8][0]), args.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(gen_result[9][0]))),
                                                           cam(tensor2numpy(gen_result[10][0]), args.img_size),
                                                           RGB2BGR(tensor2numpy(denorm(gen_result[11][0])))), 0)), 1)

                cv2.imwrite(os.path.join(args.result_dir, args.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(args.result_dir, args.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)

        if step % args.save_freq == 0:
            path = os.path.join(args.result_dir, args.dataset, 'model', '%07d' % step)
            fluid.io.save_inference_model(dirname=path, feeded_var_names=['real_A', 'real_B'], target_vars=[
                    fake_A2A_heatmap, fake_A2A, 
                    fake_A2B_heatmap, fake_A2B,
                    fake_A2B2A_heatmap, fake_A2B2A,
                    fake_B2B_heatmap, fake_B2B,
                    fake_B2A_heatmap, fake_B2A,
                    fake_B2A2B_heatmap, fake_B2A2B
            ], executor=exe, main_program=test_program)

elif args.phase == 'test':
    model_list = glob(os.path.join(args.result_dir, args.dataset, 'model/*'))
    if not len(model_list) == 0:
        model_list.sort()
        iter = int(model_list[-1].split('/')[-1])
        print(iter)
    else:
        print(" [*] Load FAILURE")
        exit(1)

    place = fluid.CUDAPlace(0) if args.device == 'cuda' else fluid.CPUPlace()
    path = os.path.join(args.result_dir, args.dataset, 'model', '%07d' % iter)

    exe = fluid.Executor(place)

    [test_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)
    print(" [*] Load SUCCESS")

    for n, data_loader in enumerate(zip(testA_loader(), testB_loader())):
        setA, setB = data_loader
        real_A, _ = zip(*setA)
        real_B, _ = zip(*setB)

        gen_result = exe.run(program=test_program, feed={feed_target_names[0]: real_A, feed_target_names[1]: real_B}, \
            fetch_list=fetch_targets)

        A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                   cam(tensor2numpy(gen_result[0][0]), args.img_size),
                                                   RGB2BGR(tensor2numpy(denorm(gen_result[1][0]))),
                                                   cam(tensor2numpy(gen_result[2][0]), args.img_size),
                                                   RGB2BGR(tensor2numpy(denorm(gen_result[3][0]))),
                                                   cam(tensor2numpy(gen_result[4][0]), args.img_size),
                                                   RGB2BGR(tensor2numpy(denorm(gen_result[5][0])))), 0)

        B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                   cam(tensor2numpy(gen_result[6][0]), args.img_size),
                                                   RGB2BGR(tensor2numpy(denorm(gen_result[7][0]))),
                                                   cam(tensor2numpy(gen_result[8][0]), args.img_size),
                                                   RGB2BGR(tensor2numpy(denorm(gen_result[9][0]))),
                                                   cam(tensor2numpy(gen_result[10][0]), args.img_size),
                                                   RGB2BGR(tensor2numpy(denorm(gen_result[11][0])))), 0)

        cv2.imwrite(os.path.join(args.result_dir, args.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)
        cv2.imwrite(os.path.join(args.result_dir, args.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)

       