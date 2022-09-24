import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import time
import datetime
import trimesh
from skimage import measure
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from loguru import logger

from utils import tensor2float, save_scalars, DictAverageMeter, SaveScene, make_nograd_func, init_net
from datasets import transforms, find_dataset_def
from models import NeuralRecon
from config import cfg, update_config
from datasets.sampler import DistributedSampler
from ops.comm import *
from models.modules import tsdf_dis,occ_dis
from torchsparse.tensor import PointTensor
from tools.evaluation_utils import *
from utils import caculate_metrics

torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

global eval_maxepoch
global eval_maxfscore
eval_maxepoch = 0
eval_maxfscore = 0

def args():
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of NeuralRecon')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        #required=True,
                        default= './config/train.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    # parse arguments and check
    args = parser.parse_args()

    return args

torch.cuda.set_device(0)
args = args()
update_config(cfg, args)
BCEloss = torch.nn.BCEWithLogitsLoss()
cfg.defrost()
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
print('number of gpus: {}'.format(num_gpus))
cfg.DISTRIBUTED = num_gpus > 1

if cfg.DISTRIBUTED:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()
cfg.LOCAL_RANK = args.local_rank
cfg.freeze()

torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed(cfg.SEED)

# create logger
if is_main_process():
    if not os.path.isdir(cfg.LOGDIR):
        os.makedirs(cfg.LOGDIR)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logfile_path = os.path.join(cfg.LOGDIR, f'{current_time_str}_{cfg.MODE}.log')
    print('creating log file', logfile_path)
    logger.add(logfile_path, format="{time} {level} {message}", level="INFO")

    tb_writer = SummaryWriter(cfg.LOGDIR)

# Augmentation
if cfg.MODE == 'train':
    n_views = cfg.TRAIN.N_VIEWS
    random_rotation = cfg.TRAIN.RANDOM_ROTATION_3D
    random_translation = cfg.TRAIN.RANDOM_TRANSLATION_3D
    paddingXY = cfg.TRAIN.PAD_XY_3D
    paddingZ = cfg.TRAIN.PAD_Z_3D
else:
    n_views = cfg.TEST.N_VIEWS
    random_rotation = False
    random_translation = False
    paddingXY = 0
    paddingZ = 0
print('cfg.MODEL.VOXEL_SIZE',cfg.MODEL.VOXEL_SIZE)
print("IsNeuralRecon ",cfg.NeuralRecon)
transform_train = []
transform_train += [transforms.ResizeImage((512,512)),
              transforms.ToTensor(),
              transforms.RandomTransformSpace(
                  cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation, random_translation,
                  paddingXY, paddingZ, max_epoch=cfg.TRAIN.EPOCHS),
              transforms.IntrinsicsPoseToProjection(n_views, 4),
              ]

transforms_train = transforms.Compose(transform_train)

transform_eval = []
transform_eval += [transforms.ResizeImage((512,512)),
              transforms.ToTensor(),
              transforms.RandomTransformSpace(
                  cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, False, False,
                  paddingXY, paddingZ, max_epoch=cfg.TRAIN.EPOCHS),
              transforms.IntrinsicsPoseToProjection(n_views, 4),
              ]

transforms_eval = transforms.Compose(transform_eval)

# dataset, dataloader
MVSDataset = find_dataset_def(cfg.DATASET)
train_dataset = MVSDataset(cfg.TRAIN.PATH, "train", transforms_train, cfg.TRAIN.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)
test_dataset = MVSDataset(cfg.TEST.PATH, "test", transforms_eval, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)

if cfg.DISTRIBUTED:
    pass
    # train_sampler = DistributedSampler(train_dataset, shuffle=False)
    # TrainImgLoader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=cfg.BATCH_SIZE,
    #     sampler=train_sampler,
    #     num_workers=cfg.TRAIN.N_WORKERS,
    #     pin_memory=True,
    #     drop_last=True
    # )
    # test_sampler = DistributedSampler(test_dataset, shuffle=False)
    # TestImgLoader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=cfg.BATCH_SIZE,
    #     sampler=test_sampler,
    #     num_workers=cfg.TEST.N_WORKERS,
    #     pin_memory=True,
    #     drop_last=False
    # )
else:
    TrainImgLoader = DataLoader(train_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TRAIN.N_WORKERS,
                                drop_last=True)
    TestImgLoader = DataLoader(test_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TEST.N_WORKERS,
                                drop_last=False)

# model, optimizer
model = NeuralRecon(cfg)
# if cfg.DISTRIBUTED:
#     model.cuda()
#     model = DistributedDataParallel(
#         model, device_ids=[cfg.LOCAL_RANK], output_device=cfg.LOCAL_RANK,
#         # this should be removed if we update BatchNorm stats
#         broadcast_buffers=False,
#         find_unused_parameters=True
#     )
# else:
#     model = torch.nn.DataParallel(model, device_ids=[0])
#     model.cuda()
model.cuda()
init_net(model)

DIS = tsdf_dis(num_classes=1, in_channels=1,
               pres=1,
               cr=1 / 2 ** 2,
               vres=1,#cfg.MODEL.VOXEL_SIZE,
               dropout=False

)
DIS.cuda()

init_net(DIS)


optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=(0.9, 0.999), weight_decay=cfg.TRAIN.WD)
optimizer_dis = torch.optim.Adam(DIS.parameters(), lr=cfg.TRAIN.LR, betas=(0.9, 0.999), weight_decay=cfg.TRAIN.WD)
eval_maxfscore = 0.0
eval_maxecpoch = 0
# main function
def train():
    # load parameters
    start_epoch = 0
    if cfg.RESUME:
        saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if len(saved_models) != 0:
            # use the latest checkpoint file
            loadckpt = os.path.join(cfg.LOGDIR, saved_models[-1])
            logger.info("resuming " + str(loadckpt))
            map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
            state_dict = torch.load(loadckpt, map_location=map_location)
            model.load_state_dict(state_dict['model'], strict=False)
            optimizer.param_groups[0]['initial_lr'] = state_dict['optimizer']['param_groups'][0]['lr']
            optimizer.param_groups[0]['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
            start_epoch = state_dict['epoch'] + 1
    elif cfg.LOADCKPT != '':
        # load checkpoint file specified by args.loadckpt
        logger.info("loading model {}".format(cfg.LOADCKPT))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
        state_dict = torch.load(cfg.LOADCKPT, map_location=map_location)
        model.load_state_dict(state_dict['model'])
        optimizer.param_groups[0]['initial_lr'] = state_dict['optimizer']['param_groups'][0]['lr']
        optimizer.param_groups[0]['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
        start_epoch = state_dict['epoch'] + 1
    logger.info("start at epoch {}".format(start_epoch))
    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    milestones = [int(epoch_idx) for epoch_idx in cfg.TRAIN.LREPOCHS.split(':')[0].split(',')]
    lr_gamma = 1 / float(cfg.TRAIN.LREPOCHS.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)
    avg_test_scalars = DictAverageMeter()

    for epoch_idx in range(start_epoch, cfg.TRAIN.EPOCHS):
        logger.info('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        TrainImgLoader.dataset.epoch = epoch_idx
        TrainImgLoader.dataset.tsdf_cashe = {}
        # training
        # 去掉train的mesh结果
        # scene = ''
        # outputs = {}
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % cfg.SUMMARY_FREQ == 0
            start_time = time.time()
            # print(scene, sample['scene'])
            # print(outputs.keys())

            #去掉train的mesh结果
            # if scene is not None and outputs.keys() is not None and scene!= sample['scene'] and 'scene_tsdf' in outputs.keys():
            #     metircs = caculate_metrics(outputs['scene_tsdf'], outputs['scene_tsdf_target'], outputs['origin'],
            #                                outputs['origin_target'], cfg.MODEL.VOXEL_SIZE, sample['scene'],
            #                                epoch_idx=epoch_idx, mode = 'train')
            #
            #     logger.info(
            #         'Train: Epoch {}/{}, sence = {}, fscore = {:.3f}, acc = {:.3f}, recall = {:.3f}'.format(
            #             epoch_idx, cfg.TRAIN.EPOCHS, scene, metircs['fscore'], metircs['prec'], metircs['recal']))
            if (batch_idx+1) % 9==0 and batch_idx!=0:
                save_sence = True
            else:
                save_sence = False
            loss, scalar_outputs, dis_loss, gen_loss, outputs = train_sample(sample,batch_idx,save_sence= save_sence)#False)
            ##去掉train的mesh结果
            # scene = sample['scene']
            if is_main_process():
                logger.info(
                    'Epoch {}/{}, Iter {}/{}, sence:{}, train loss = {:.3f},time = {:.3f}'.format(epoch_idx, cfg.TRAIN.EPOCHS,
                                                                                         batch_idx,
                                                                                         len(TrainImgLoader), sample['scene'], loss,
                                                                                         time.time() - start_time))
            if loss!=0:
                avg_test_scalars.update(scalar_outputs)
            del scalar_outputs

        # checkpoint
        if (epoch_idx + 1) % cfg.SAVE_FREQ == 0 and is_main_process():
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(cfg.LOGDIR, epoch_idx))

        loss_mean = avg_test_scalars.mean()
        # print(loss_mean)
        if is_main_process():
            logger.info(
                'Epoch {}/{}, total_loss = {:.3f}'.format(
                    epoch_idx, cfg.TRAIN.EPOCHS, loss_mean['total_loss']))
        # if "scene_tsdf" in outputs.keys():
        #     metircs = caculate_metrics(outputs['scene_tsdf'], outputs['scene_tsdf_target'], outputs['origin'],
        #                                outputs['origin_target'], cfg.MODEL.VOXEL_SIZE, sample['scene'],threshold= 1.5,
        #                                epoch_idx=epoch_idx, mode = 'train')

            # logger.info(
            #     'Train: Epoch {}/{}, sence = {}, fscore = {:.3f}, acc = {:.3f}, recall = {:.3f}'.format(
            #         epoch_idx, cfg.TRAIN.EPOCHS, "scene", metircs['fscore'], metircs['prec'], metircs['recal']))

        # if do_summary and is_main_process():
        # print("write logg into tensobaord")
        save_scalars(tb_writer, 'train', loss_mean, metrics=None, epoch_idx = epoch_idx)
        avg_test_scalars.reset()
        eval(epoch_idx)
        model.reset_volume()
def eval(epoch_idx):

    global eval_maxfscore
    global eval_maxepoch
    metircs_scalars = DictAverageMeter()
    avg_test_scalars = DictAverageMeter()
    # TestImgLoader.dataset.tsdf_cashe = {}
    logger.info('evaling...')
    TestImgLoader.dataset.epoch = epoch_idx
    TestImgLoader.dataset.tsdf_cashe = {}
    save_mesh_scene = SaveScene(cfg)
    batch_len = len(TestImgLoader)
    for batch_idx, sample in enumerate(TestImgLoader):
        # for n in sample['fragment']:
        #     logger.info(n)
        # save mesh if SAVE_SCENE_MESH and is the last fragment
        save_scene = cfg.SAVE_SCENE_MESH and (batch_idx+1)%9==0

        start_time = time.time()
        # print(save_scene)
        loss, scalar_outputs, outputs = test_sample(sample, save_scene)

        tsdf_vol = outputs["tsdf"]
        # print(torch.max(tsdf_vol), torch.min(tsdf_vol), torch.mean(tsdf_vol))
        # verts, faces, norms, vals = measure.marching_cubes(tsdf_vol)
        # verts = verts * 1.0 + [0,0,0]  # voxel grid coordinates to world coordinates
        # mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)



        logger.info('Epoch {}, Iter {}/{}, test loss = {:.3f}, secene = {},time = {:3f}'.format(epoch_idx, batch_idx,
                                                                                    len(TestImgLoader),
                                                                                    loss,sample['scene'],
                                                                                    time.time() - start_time))
        if loss != 0:
            avg_test_scalars.update(scalar_outputs)
        del scalar_outputs
        # print(outputs.keys(), (batch_idx+1)%9==0 )
        if 'scene_tsdf' in outputs.keys(): #and (batch_idx+1)%11==0:
            metircs = caculate_metrics(outputs['scene_tsdf'], outputs['scene_tsdf_target'], outputs['origin'],
                                       outputs['origin_target'], cfg.MODEL.VOXEL_SIZE, sample['scene'], threshold=1.5,
                                       epoch_idx=epoch_idx, mode='eval')

            logger.info(
                'EVAL: Epoch {}/{}, scene = {}, fscore = {:.3f}, acc = {:.3f}, recall = {:.3f}'.format(
                    epoch_idx, cfg.TRAIN.EPOCHS, sample['scene'], metircs['fscore'], metircs['prec'], metircs['recal']))

            metircs_scalars.update(metircs)
            del metircs

        # if batch_idx % 100 == 0:
        #     logger.info("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader),
        #                                                        avg_test_scalars.mean()))

        # save mesh
        # if cfg.SAVE_SCENE_MESH:
        #     save_mesh_scene(outputs, sample, epoch_idx)

    loss_mean = avg_test_scalars.mean()
    logger.info(
        'EVAL: Epoch {}/{}, total_loss = {:.3f}'.format(
            epoch_idx, cfg.TRAIN.EPOCHS, loss_mean['total_loss']))
    metircs_mean = metircs_scalars.mean()
    if 'fscore' in metircs_mean.keys():
        if metircs_mean['fscore'] > eval_maxfscore:
            eval_maxfscore = metircs_mean['fscore']
            eval_maxepoch = epoch_idx
        #eval_maxepoch = 0
        save_scalars(tb_writer, 'fulltest', avg_test_scalars.mean(), metircs_mean, epoch_idx)
        logger.info(
            'EVAL: Epoch {}/{}, fscore_mean = {:.3f}, acc_mean = {:.3f}, recall_mean = {:.3f}'.format(
                epoch_idx, cfg.TRAIN.EPOCHS, metircs_mean['fscore'], metircs_mean['prec'], metircs_mean['recal']))
        logger.info(
            'Best fscor: {} in epoch {}'.format(
                eval_maxfscore, eval_maxepoch))
    else:
        save_scalars(tb_writer, 'fulltest', avg_test_scalars.mean(), epoch_idx = epoch_idx)
    metircs_scalars.reset()
    avg_test_scalars.reset()


def test(from_latest=False):
    ckpt_list = []
    while True:
        saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        if from_latest:
            saved_models = saved_models[-1:]
        for ckpt in saved_models:
            if ckpt not in ckpt_list:
                # use the latest checkpoint file
                loadckpt = os.path.join(cfg.LOGDIR, ckpt)
                logger.info("resuming " + str(loadckpt))
                state_dict = torch.load(loadckpt)
                model.load_state_dict(state_dict['model'])
                epoch_idx = state_dict['epoch']

                # TestImgLoader.dataset.tsdf_cashe = {}

                avg_test_scalars = DictAverageMeter()
                save_mesh_scene = SaveScene(cfg)
                batch_len = len(TestImgLoader)
                for batch_idx, sample in enumerate(TestImgLoader):
                    for n in sample['fragment']:
                        logger.info(n)
                    # save mesh if SAVE_SCENE_MESH and is the last fragment
                    save_scene = cfg.SAVE_SCENE_MESH and batch_idx == batch_len - 1

                    start_time = time.time()
                    loss, scalar_outputs, outputs = test_sample(sample, save_scene)
                    logger.info('Epoch {}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, batch_idx,
                                                                                                len(TestImgLoader),
                                                                                                loss,
                                                                                                time.time() - start_time))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs

                    if batch_idx % 100 == 0:
                        logger.info("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader),
                                                                           avg_test_scalars.mean()))

                    # save mesh
                    if cfg.SAVE_SCENE_MESH:
                        save_mesh_scene(outputs, sample, epoch_idx)



                metircs = caculate_metrics(outputs['scene_tsdf'], outputs['scene_tsdf_target'], outputs['origin'],
                                           outputs['origin_target'], cfg.MODEL.VOXEL_SIZE, sample['scene'], epoch_idx=epoch_idx)
                logger.info('Epoch {}, fscore = {:.3f}, acc = {:.3f}, recall = {:.3f}'.format(
                                    epoch_idx, metircs['fscore'], metircs['prec'], metircs['recal']))

                save_scalars(tb_writer, 'fulltest', avg_test_scalars.mean(), metircs, epoch_idx)
                logger.info("epoch {} avg_test_scalars:".format(epoch_idx), avg_test_scalars.mean())

                ckpt_list.append(ckpt)

        time.sleep(10)

# metircs = {'dist1': 0.0,
#            'dist2': 0.0,
#            'prec': 0.0,
#            'recal': 0.0,
#            'fscore': 0.0,
#            }
# if "scene_tsdf" in global_mesh.keys():
#     metircs = caculate_metrics(global_mesh['scene_tsdf'], global_mesh['scene_tsdf_target'],
#                                global_mesh['origin'], global_mesh['origin_target'],
#                                cfg.MODEL.VOXEL_SIZE, sample['scene'], epoch_idx=epoch_idx)
#     if is_main_process():
#         logger.info(
#             'Epoch {}/{}, fscore = {:.3f}, acc = {:.3f}, recall = {:.3f}'.format(
#                 epoch_idx, cfg.TRAIN.EPOCHS, metircs['fscore'], metircs['prec'], metircs['recal']))




def train_sample(sample,batch_idx, save_sence):
    torch.autograd.set_detect_anomaly(True)
    ### train tsdf genrate network
    model.train()
    optimizer.zero_grad()

    outputs, loss_dict = model(sample, save_sence)
    # print(loss_dict['julei_loss'])
    loss = loss_dict['total_loss']
    gen_loss = 0.0
    dis_loss = 0.0
    if cfg.Need_DIS and ('tsdf' in outputs.keys()):
        fake_data = PointTensor(outputs['tsdf'], outputs['coords'])
        real_data = PointTensor(outputs['tsdf_target'], outputs['coords'])
        result = DIS(fake_data)
        real_label = torch.ones(result.shape).cuda()
        gen_loss = BCEloss(result.permute(1, 0),real_label.permute(1, 0))
        total_loss = 2*loss + gen_loss
        loss_dict['gen_loss'] = gen_loss
    else:
        total_loss = loss
    # print('loss',total_loss)
    total_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    ##train tsdf discramater network
    if batch_idx %3==0 and cfg.Need_DIS and ('tsdf' in outputs.keys()):
        DIS.train()
        optimizer_dis.zero_grad()
        result = DIS(real_data)
        real_loss = BCEloss(result.permute(1, 0),real_label.permute(1, 0))

        result = DIS(fake_data.detach())
        fake_label = torch.zeros(result.shape).cuda()
        fake_loss = BCEloss(result.permute(1, 0),fake_label.permute(1, 0))

        dis_loss = (real_loss + fake_loss) /2
        dis_loss.backward()
        torch.nn.utils.clip_grad_norm_(DIS.parameters(), 1.0)
        optimizer_dis.step()
        return tensor2float(loss), tensor2float(loss_dict), tensor2float(dis_loss), tensor2float(gen_loss), outputs
    return tensor2float(loss), tensor2float(loss_dict), tensor2float(dis_loss),tensor2float(gen_loss), outputs

@make_nograd_func
def test_sample(sample, save_scene=True):
    model.eval()

    outputs, loss_dict = model(sample, save_scene)
    loss = loss_dict['total_loss']

    return tensor2float(loss), tensor2float(loss_dict), outputs


if __name__ == '__main__':
    if cfg.MODE == "train":
        train()
    # elif cfg.MODE == "test":
    #     test(True)
