import os, sys
import logging
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from medpy.metric import dc

from tensorboardX import SummaryWriter
from datasets.dataset_acdc import ACDCdataset, ACDCdatasetFast, RandomGenerator
from networks import CENet
from utils import print_param_flops, plot_result, get_optimizer, get_scheduler, Criterion
from utils.utils_acdc import inference


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=20, help='batch_size per gpu')
parser.add_argument("--save_path", default="./model_pth/ACDC") # Check the root Dir: SAVE path (checked)
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", type=str, default="", help="path to the checkpoint file to resume training or evaluate")
parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="./data/ACDC/") # Check the root Dir: Dataset root (Aval)
parser.add_argument("--volume_path", default="./data/ACDC/test") # Check the root Dir: test root (Aval)
parser.add_argument("--z_spacing", default=10)
parser.add_argument('--input_channels', type=int, default=1, help='input channels of network input')
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!') # Check the test_save Dir: Preds root
parser.add_argument("--model_name", type=str, default="cenet", help="model_name")
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer: [SGD, AdamW, Adam])')
parser.add_argument('--scheduler', type=str, default='poly', help='scheduler: [cosine, step, poly, exp, custom]')
parser.add_argument('--max_epochs', type=int, default=200, help='maximum epoch number to train')
parser.add_argument('--num_workers', type=int, default=12, help='num_workers')
parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
parser.add_argument('--scale_factors', type = str, default = "0.8,0.4", help = "Boundary enhancement downsample scale factors")
parser.add_argument('--num_heads', type=str, default="2,2,2", help='number of heads in each layer. first is bigger')
parser.add_argument('--concatenation', action='store_true', default=False, help='use this flag to concatenate feature maps in MSDC block')
parser.add_argument('--encoder', type=str, default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
parser.add_argument('--freeze_bb', action='store_true', default=False, help='use this flag to freeze backbone weights')
parser.add_argument('--no_ptenc', action='store_true', default=False, help='use this flag to turn off loading pretrained enocder weights')
parser.add_argument('--base_lr', type=float,  default=0.05, help='segmentation network learning rate')
parser.add_argument('--use_chn_decompose', action='store_true', help = "use moga-based channel aggerigtion")
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--amp', action='store_true', help='AMP mode')
parser.add_argument('--fast_data', action='store_true', help='FastDataset')
parser.add_argument('--skip_mode', type=str, default="cat", choices=["cat", "add"], help='use this flag to determine the mode of input for skip enhancement module')
parser.add_argument('--loss_type', type=str, default='boundary', help='loss function type [ce, boundary, dice]')
parser.add_argument('--loss_weights', type=str, default='1', help='loss weights for different losses ["1,1,0.5"]')
parser.add_argument('--compile', action='store_true', help='Compile the model')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--dec_up_block', type=str, default="eucb", choices=["uprb", "eucb", "upcn", "uptc"], help="uprb: upsample with residual block, eucb: eca upsample block, upcn: upsample with conv block, uptc: upsample with transpose conv block")
parser.add_argument('--encoder_ptdir', type=str, default='.', help='base path to pretrained encoder weights')
parser.add_argument('--out_merge_mode', type=str, default="cat", choices=["cat", "add"], help="cat or add")
parser.add_argument('--out_up_block', type=str, default="upcn", choices=["uprb", "eucb", "upcn", "uptc"], help="uprb: upsample with residual block, eucb: eca upsample block, upcn: upsample with conv block, uptc: upsample with transpose conv block")
parser.add_argument('--out_up_ks', type=int, default=3, choices=[1, 3, 5], help="[1, 3, 5] -> upsample kernel size")


args = parser.parse_args()

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


snapshot_path = f"{args.save_path}/{args.tag}"
snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_' + str(args.img_size)
snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
test_save_path = os.path.join(args.test_save_dir, args.tag)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True)

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
writer = SummaryWriter(snapshot_path + '/log')

_log_fn = "eval"if args.eval else "train"
logging.basicConfig(
    filename=snapshot_path + "/log_" + _log_fn + ".txt", 
    level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
if args.eval:
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(str(args))
log_filename = f'{snapshot_path}' + '/log_' + _log_fn + '.txt'


from pprint import pprint
pprint(vars(args))
net = CENet(
    input_channels=args.input_channels,
    num_classes=args.num_classes, 
    scale_factors = [float(s) for s in args.scale_factors.split(',')], 
    encoder=args.encoder,
    enc_pretrain=not args.no_ptenc and not args.eval,  # do not load pretrained weights in eval mode
    freeze_bb=args.freeze_bb,
    skip_mode=args.skip_mode,
    diffatt_num_heads=[int(h) for h in args.num_heads.split(',')],
    dec_up_block=args.dec_up_block,
    out_merge_mode=args.out_merge_mode,
    out_up_block=args.out_up_block,
    out_up_ks=args.out_up_ks,
    base_ptdir=args.encoder_ptdir,
).cuda()

print_param_flops(net, args)

# by EMCAD paper
from utils import CalParams
CalParams(net, torch.zeros((1, args.input_channels, args.img_size, args.img_size)).cuda())


DatasetClass = ACDCdatasetFast if args.fast_data else ACDCdataset
db_train = DatasetClass(args.root_dir, args.list_dir, split="train", 
                       transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
db_val = DatasetClass(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
db_test = DatasetClass(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
tr_loader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True)
vl_loader = DataLoader(db_val, batch_size=1, shuffle=False)
te_loader = DataLoader(db_test, batch_size=1, shuffle=False)

print(f"The length of train set is: {len(db_train)}")
print(f"The length of val set is: {len(db_val)}")
print(f"The length of test set is: {len(db_test)}")


if args.eval:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluation mode")
    net.eval()
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file {args.checkpoint} does not exist.")
        net.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    elif os.path.exists(os.path.join(snapshot_path)):
        print(f"Loading best weights from {os.path.join(snapshot_path, 'best.pth')}")
        net.load_state_dict(torch.load(os.path.join(snapshot_path, 'best.pth'), weights_only=True))
    else:
        print("No weights file provided...")
        exit(0)
    test_save_dir = os.path.join(snapshot_path, "predictions")
    os.makedirs(test_save_dir, exist_ok=True)
    te_avg_dcs, te_avg_hd = inference(args, net, te_loader, test_save_dir, -1)
    print(f"ACDC -> Test <{args.tag}> -> Average Dice: {te_avg_dcs:.4f}, Average HD: {te_avg_hd:.4f}")
    exit(0)


if args.checkpoint:
    print(f"Loading checkpoint from {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file {args.checkpoint} does not exist.")
    net.load_state_dict(torch.load(args.checkpoint, weights_only=True))


if args.n_gpu > 1:
    net = nn.DataParallel(net)
net = net.cuda()


# training mode
print("Training mode")
net.train()

criterion = Criterion(args.num_classes, args)
if args.compile: 
    print("Compiling the model...")
    net = torch.compile(net, mode='default', fullgraph=True)
    criterion = torch.compile(criterion)
if args.amp:
    try:
        from torch.amp import autocast, GradScaler
    except ImportError:
        from torch.amp.autocast_mode import autocast
        from torch.amp.grad_scaler import GradScaler
    scaler = GradScaler()
    print("AMP enabled...")
else:
    print("AMP disabled...")

iter_num = 0

Loss = []
te_accuracy = []
best_dcs_vl = 0
best_dcs_te = 0
dice_ = []
hd95_ = []

max_iterations = args.max_epochs * len(tr_loader)
writer = SummaryWriter(snapshot_path + '/log')

optimizer = get_optimizer(net, args)
scheduler = get_scheduler(optimizer, args, max_iterations=args.max_epochs*len(tr_loader))

def val():
    logging.info("Validation ===>")
    dc_sum = 0
    net.eval()
    for i, val_sampled_batch in enumerate(vl_loader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda().unsqueeze(1)
        val_outputs = net(val_image_batch)
        val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
        dc_sum += dc(val_outputs.cpu().data.numpy(), val_label_batch[:].cpu().data.numpy())
    performance = dc_sum / len(vl_loader)
    logging.info('Testing performance in val model) mean_dice:%f, best_dice:%f' % (performance, best_dcs_vl))
    return performance


for epoch in range(0, args.max_epochs):
    net.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(tr_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        optimizer.zero_grad()
        if args.amp:
            with autocast(device_type='cuda'):
                outputs = net(image_batch)
                loss = criterion(outputs, label_batch[:])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = net(image_batch)         
            loss = criterion(outputs, label_batch[:])
            loss.backward()
            optimizer.step()

        lr_ = scheduler.get_last_lr()[0]
        scheduler.step()

        iter_num = iter_num + 1
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/criterion', loss, iter_num)

        if iter_num % 20 == 0:
            logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()
    
    Loss.append(train_loss / len(db_train))
    logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))

    vl_avg_dcs = val()

    if vl_avg_dcs >= best_dcs_vl:
        te_avg_dcs, te_avg_hd = inference(args, net, te_loader, args.test_save_dir, epoch+1)
        if te_avg_dcs >= best_dcs_te:
            best_dcs_vl = vl_avg_dcs
            best_dcs_te = te_avg_dcs
            save_model_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(net.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
        dice_.append(te_avg_dcs)
        hd95_.append(te_avg_hd)
        te_accuracy.append(te_avg_dcs)

    print(f"epoch:{epoch:03d}/{args.max_epochs}, loss:{train_loss/len(db_train):0.5f}, lr:{lr_:0.6f}, vl_DCS:{vl_avg_dcs*100:0.3f}, te_DCS:{te_avg_dcs*100:0.3f}, te_HD95:{te_avg_hd:0.2f}")

    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(snapshot_path, 'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, te_avg_dcs))
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        break

plot_result(dice_, hd95_, snapshot_path, args)
writer.close()
