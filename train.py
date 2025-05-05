import os
import sys
import time
import logging
import argparse
import numpy as np
import utils
from PIL import Image
import torchvision
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import math
from xdecoder.infer_semseg import segment
from dataset.dataset_teacher_FMB import Data as Data_Teacher
from model_main.model import Network as Network_Teacher
from model_main.segment_loss import process_input, calculate_loss

# from dataset_student.train_dataset_Data import Data as Data_Student
from model_sub.model import Network as Network_Student
from model_sub.losses import Losses

# from seg_core.model import WeTr
# from seg_utils.optimizer import PolyWarmupAdamW
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser("ruas")
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--crop_size', type=tuple, default=(192, 256))

parser.add_argument('--root_dir', type=str, default='./data/train', help='location of the datasets')
parser.add_argument('--use_mask_num', type=int, default=4)
parser.add_argument('--cache_mask_num', type=int, default=20)
# cache_mask_num 决定缓存多少个掩码，use_mask_num 决定使用多少个掩码
parser.add_argument('--checkpoint_teacher', type=str, default='', help='location of the checkpoint')
parser.add_argument('--checkpoint_student', type=str, default='', help='location of the checkpoint')



parser.add_argument('--learning_rate_Teacher', type=float, default=0.0005, help='init learning rate')
parser.add_argument('--learning_rate_Student', type=float, default=0.002, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-5) #0.000000001, help='min learning rate')

parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--test_freq', type=float, default=10, help='report frequency')

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')

parser.add_argument('--ini_epoch', type=str, default=1, help='segment start epoch')
parser.add_argument('--student_epoch', type=int, default=1, help='student start epoch')




args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# if args.test_freq > args.epochs:
# 	args.test_freq = args.epochs

def main():
	setup_experiment_directories(args)
	if not torch.cuda.is_available():
		teacher_logger.info('no gpu device available')
		sys.exit(1)

	np.random.seed(args.seed)
	cudnn.benchmark = True
	torch.manual_seed(args.seed)
	cudnn.enabled = True
	torch.cuda.manual_seed(args.seed)
	teacher_logger.info("args = %s", args)
	student_logger.info("args = %s", args)


	train_data = Data_Teacher(mode='train', use_mask_num=args.use_mask_num, cache_mask_num=args.cache_mask_num, crop_size=args.crop_size, root_dir=args.root_dir)# crop resize flip

	train_data_size = len(train_data)
	indices = list(range(train_data_size))

	train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
											  pin_memory=True, num_workers=4, drop_last=False)

	teacher_logger.info("train_data_size: {}".format(train_data_size))
	student_logger.info("train_data_size: {}".format(train_data_size))

	device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
   
	model_teacher = Network_Teacher(mask_num=args.use_mask_num)
	model_student = Network_Student()

	

	teacher_logger.info('lr = {}'.format(args.learning_rate_Teacher))
	student_logger.info('lr = {}'.format(args.learning_rate_Student))

	if args.checkpoint_teacher!='':
		teacher_logger.info('loading {}'.format(args.checkpoint_teacher))
		model_teacher.load_state_dict(torch.load(args.checkpoint_teacher))
	if args.checkpoint_student!='':
		student_logger.info('loading {}'.format(args.checkpoint_student))
		model_student.load_state_dict(torch.load(args.checkpoint_student))

	model_teacher.to(device)
	model_student.to(device)   

	optimizer_teacher = torch.optim.Adam(model_teacher.parameters(), lr=args.learning_rate_Teacher)

	scheduler_teacher = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_teacher, float(args.epochs)*len(train_loader), eta_min=args.learning_rate_min)
	# scheduler_teacher = torch.optim.lr_scheduler.StepLR(optimizer_teacher, step_size=args.student_epoch, gamma = target_lr / args.learning_rate_Teacher)
	
	optimizer_student = torch.optim.Adam(model_student.parameters(), lr=args.learning_rate_Student)

	scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_student, float(args.epochs)*len(train_loader), eta_min=args.learning_rate_min)
	
	# teacher_logger.info("param size = %fMB", utils.count_parameters_in_MB(model_teacher))
	
	# teacher_logger.info('********** < model constructed > **********')

	student_logger.info("param size = %fMB", utils.count_parameters_in_MB(model_student))
	
	student_logger.info('********** < model constructed > **********')

	train(model_teacher, model_student, optimizer_teacher, optimizer_student, scheduler_teacher, scheduler_student, device, train_loader)



def train(model_teacher, model_student, optimizer_teacher, optimizer_student, scheduler_teacher, scheduler_student, device, train_loader):
	writer_teacher = SummaryWriter(os.path.join(inference_dir_Teacher, 'train_log'))
	input = torch.ones((1, 1, args.crop_size[0], args.crop_size[1])).to(device)
	input_2 = torch.ones((1, args.use_mask_num, args.crop_size[0], args.crop_size[1])).to(device) 
	writer_teacher.add_graph(model_teacher, (input, input, input_2, input_2))
	del input
	del input_2

	writer_student = SummaryWriter(os.path.join(inference_dir_Student, 'train_log'))
	input = torch.ones((1, 1, args.crop_size[0], args.crop_size[1])).to(device)
	writer_student.add_graph(model_student, (input, input))
	del input

	total_train_step_teacher = 0
	total_train_step_student = 0

	for i in range(1, args.epochs+1):
		teacher_logger.info("--------------- epoch {} lr {} ---------------".format(i, scheduler_teacher.get_lr()[0]))
		student_logger.info("--------------- epoch {} lr {} ---------------".format(i, scheduler_student.get_lr()[0]))

		model_teacher.train()
		model_student.train()

		total_fuse_loss_teacher = 0
		total_grad_loss_teacher = 0
		total_train_loss_teacher = 0

		step_teacher= 0
		step_student= 0

		total_fuse_loss_student = 0
		total_grad_loss_student = 0
		total_train_loss_student = 0
		total_contrast_loss_student = 0
		total_DHs_student = [0] * 2

		losses = Losses()

		
		train_iter = iter(train_loader)
		batch_idx = 0
		
		# 记录上一个批次的数据，以便在需要时重用
		last_batch_data = None
		
		while True:
			try:
				# 获取当前批次数据
				current_batch_data = next(train_iter)
				
				# 如果学生网络已开始训练，交替使用批次
				if i >= args.student_epoch:
					# 偶数批次用于学生网络，奇数批次用于教师网络
					if batch_idx % 2 == 0:
						# 学生网络处理
						student_losses = process_student_batch(
							current_batch_data, model_teacher, model_student, 
							optimizer_student, scheduler_student, losses, device,
							writer_student, total_train_step_student, step_student,
							total_fuse_loss_student, total_grad_loss_student, 
							total_contrast_loss_student, total_DHs_student, total_train_loss_student
						)
						
						# 更新累计损失值
						total_fuse_loss_student = student_losses[0]
						total_grad_loss_student = student_losses[1]
						total_contrast_loss_student = student_losses[2]
						total_DHs_student = student_losses[3]
						total_train_loss_student = student_losses[4]
						
						step_student += 1
						total_train_step_student += 1
					else:
						# 教师网络处理
						teacher_losses = process_teacher_batch(
							current_batch_data, model_teacher, model_student, 
							optimizer_teacher, scheduler_teacher, losses, device, i,
							writer_teacher, total_train_step_teacher, step_teacher,
							total_fuse_loss_teacher, total_grad_loss_teacher, total_train_loss_teacher
						)
						
						# 更新累计损失值
						total_fuse_loss_teacher = teacher_losses[0]
						total_grad_loss_teacher = teacher_losses[1]
						total_train_loss_teacher = teacher_losses[2]
						
						step_teacher += 1
						total_train_step_teacher += 1
				else:
					# 在学生网络训练前，所有批次都用于教师网络
					teacher_losses = process_teacher_batch(
						current_batch_data, model_teacher, model_student, 
						optimizer_teacher, scheduler_teacher, losses, device, i,
						writer_teacher, total_train_step_teacher, step_teacher,
						total_fuse_loss_teacher, total_grad_loss_teacher, total_train_loss_teacher
					)
					
					# 更新累计损失值
					total_fuse_loss_teacher = teacher_losses[0]
					total_grad_loss_teacher = teacher_losses[1]
					total_train_loss_teacher = teacher_losses[2]
					
					step_teacher += 1
					total_train_step_teacher += 1
				
				# 保存当前批次作为上一个批次
				last_batch_data = current_batch_data
				batch_idx += 1
				
			except StopIteration:
				# 当迭代器结束时
				if i >= args.student_epoch and batch_idx % 2 == 0 and last_batch_data is not None:
					
					teacher_logger.info("Reusing last batch for teacher at end of epoch")
					teacher_losses = process_teacher_batch(
						last_batch_data, model_teacher, model_student, 
						optimizer_teacher, scheduler_teacher, losses, device, i,
						writer_teacher, total_train_step_teacher, step_teacher,
						total_fuse_loss_teacher, total_grad_loss_teacher, total_train_loss_teacher
					)
					
					# 更新累计损失值
					total_fuse_loss_teacher = teacher_losses[0]
					total_grad_loss_teacher = teacher_losses[1]
					total_train_loss_teacher = teacher_losses[2]
					
					step_teacher += 1
					total_train_step_teacher += 1
				break

		# 保存模型检查点
		if i >= args.student_epoch:
			torch.save(model_student.state_dict(), os.path.join(model_path_Student, 'epoch_{}.pt'.format(i)))
			student_logger.info('saving epoch {} model'.format(i))
		
		torch.save(model_teacher.state_dict(), os.path.join(model_path_Teacher, 'epoch_{}.pt'.format(i)))
		teacher_logger.info('saving epoch {} model'.format(i))

		# 记录每个 epoch 的总损失
		teacher_logger.info("Teacher:epoch {}: total_fuse_loss: {}, total_grad_loss: {}, total_train_loss: {}".format(
			i, total_fuse_loss_teacher, total_grad_loss_teacher, total_train_loss_teacher))
		writer_teacher.add_scalar("total_fuse_loss", total_fuse_loss_teacher, i)
		writer_teacher.add_scalar("total_grad_loss", total_grad_loss_teacher, i)
		writer_teacher.add_scalar("total_train_loss", total_train_loss_teacher, i)

		if i >= args.student_epoch:
			student_logger.info("Student:epoch {}: total_fuse_loss: {}, total_grad_loss: {}, total_contrast_loss: {}, total_train_loss: {}".format(
				i, total_fuse_loss_student, total_grad_loss_student, total_contrast_loss_student, total_train_loss_student))
			student_logger.info("Student:total_DHs: {}".format(total_DHs_student))
			writer_student.add_scalar("total_fuse_loss", total_fuse_loss_student, i)
			writer_student.add_scalar("total_grad_loss", total_grad_loss_student, i)
			writer_student.add_scalar("total_contrast_loss", total_contrast_loss_student, i)
			writer_student.add_scalar("total_train_loss", total_train_loss_student, i)

	writer_teacher.close()
	writer_student.close()

def process_student_batch(data, model_teacher, model_student, optimizer_student, scheduler_student, 
						  losses, device, writer_student, total_train_step_student, step_student,
						  total_fuse_loss_student, total_grad_loss_student, total_contrast_loss_student, 
						  total_DHs_student, total_train_loss_student):
	"""处理学生网络的一个批次"""
	names, ir_mask, vi_mask, label, ir, y, cb, cr, label_mask = data.values()
	ir_mask, vi_mask, ir, y, cb, cr, label_mask = utils.togpu_7(device, ir_mask, vi_mask, ir, y, cb, cr, label_mask)
	
	# 1. 计算教师网络的输出(用于指导学生网络)
	with torch.no_grad():  
		output_teacher, intermediate_outputs_teacher = model_teacher(y, ir, vi_mask, ir_mask)
	
	# 2. 计算学生网络的输出
	output_student, intermediate_outputs_student = model_student(y, ir)

	# 3. 计算学生网络的损失
	loss_student, fuse_loss_student, loss_grad_student, contrast_loss_student, DH_value_student = losses.cal(
		output_student, y, ir, output_teacher, vi_mask, ir_mask
	)
	loss_middle = 0.3*calculate_cosine_similarity_loss(intermediate_outputs_student, intermediate_outputs_teacher)
	total_loss_student = loss_student + loss_middle

	# 打印学生网络的损失信息
	print('Student:lr:', scheduler_student.get_last_lr()[0], 'fuse_loss: ', fuse_loss_student.item(),
		'grad_loss: ', loss_grad_student.item(), 'contrast_loss: ', contrast_loss_student.item(),
		'loss: ', total_loss_student.item())
	print('Student:DH_value:', DH_value_student)

	if not math.isfinite(total_loss_student.item()):
		student_logger.info("Loss is {}, stopping training".format(total_loss_student.item()))
		sys.exit(1)

	# 4. 更新学生网络参数
	optimizer_student.zero_grad()
	total_loss_student.backward()
	optimizer_student.step()
	scheduler_student.step()

	# 更新累计损失
	total_fuse_loss_student += fuse_loss_student.item()
	total_grad_loss_student += loss_grad_student.item()
	total_contrast_loss_student += contrast_loss_student.item()
	for idx2 in range(len(total_DHs_student)):
		total_DHs_student[idx2] += DH_value_student[idx2]
	total_train_loss_student += loss_student.item()

	# 记录和可视化
	if total_train_step_student % args.report_freq == 0 or total_train_step_student == 1:
		report_path = EXP_path_Student

		output_colored = utils.YCrCb2RGB(torch.cat((output_student, cb, cr), dim=1))
		ref_colored = utils.YCrCb2RGB(torch.cat((output_teacher, cb, cr), dim=1))
		input_vis = utils.YCrCb2RGB(torch.cat((y, cb, cr), dim=1))
		y2rgb = torch.cat((y, y, y), dim=1)

		torchvision.utils.save_image(input_vis, os.path.join(report_path, 'vis.png'))
		torchvision.utils.save_image(y2rgb, os.path.join(report_path, 'y.png'))
		torchvision.utils.save_image(ir, os.path.join(report_path, 'ir.png'))
		torchvision.utils.save_image(output_student, os.path.join(report_path, 'output.png'))
		torchvision.utils.save_image(output_colored, os.path.join(color_path_Student,  'output_color.png'))
		torchvision.utils.save_image(output_colored, os.path.join(color_path_Student,  f'output_colored{total_train_step_student}.png'))
		torchvision.utils.save_image(output_teacher, os.path.join(report_path, 'ref.png'))
		torchvision.utils.save_image(ref_colored, os.path.join(report_path, 'ref_colored.png'))

		if not os.path.exists(os.path.join(report_path, 'masks')):
			os.makedirs(os.path.join(report_path, 'masks'))                            
		y_mask = torch.split(vi_mask, 1, dim=1)
		for idx2, y_m in enumerate(y_mask):
			torchvision.utils.save_image(y_m, os.path.join(report_path, 'masks', 'vi_{}.png'.format(idx2)))

		ir_mask1 = torch.split(ir_mask, 1, dim=1)
		for idx2, ir_m in enumerate(ir_mask1):
			torchvision.utils.save_image(ir_m, os.path.join(report_path, 'masks', 'ir_{}.png'.format(idx2)))
		
		student_logger.info("Student:step: {}, total_step: {}, lr: {}, fuse_loss: {}, grad loss: {}, contrast loss: {}, loss: {}".format(
			step_student, total_train_step_student, scheduler_student.get_last_lr()[0], 
			fuse_loss_student.item(), loss_grad_student.item(), 
			contrast_loss_student.item(), loss_student.item()))
		student_logger.info("DH_value: {}".format(DH_value_student))
		
		writer_student.add_scalar("fuse_loss", fuse_loss_student.item(), total_train_step_student)
		writer_student.add_scalar("grad_loss", loss_grad_student.item(), total_train_step_student)
		writer_student.add_scalar("contrast_loss", contrast_loss_student.item(), total_train_step_student)
		for idx2 in range(len(DH_value_student)):
			writer_student.add_scalar("DH_value_{}".format(idx2), DH_value_student[idx2], total_train_step_student)
		writer_student.add_scalar("train_loss", loss_student.item(), total_train_step_student)
		writer_student.add_images("vis", input_vis, total_train_step_student)
		writer_student.add_images("y", y2rgb, total_train_step_student)
		writer_student.add_images("ir", ir, total_train_step_student)
		writer_student.add_images("output_colored", output_colored, total_train_step_student)
		writer_student.add_images("ref_colored", ref_colored, total_train_step_student)
		print(names)

	# 保存模型检查点
	if total_train_step_student % args.test_freq == 0:
		torch.save(model_student.state_dict(), os.path.join(model_path_Student, '{}.pt'.format(total_train_step_student)))
		student_logger.info('saving pt {}'.format(total_train_step_student))
	
	# 清理内存
	del output_teacher, intermediate_outputs_teacher, output_student, intermediate_outputs_student
	torch.cuda.empty_cache()
	
	return total_fuse_loss_student, total_grad_loss_student, total_contrast_loss_student, total_DHs_student, total_train_loss_student

def process_teacher_batch(data, model_teacher, model_student, optimizer_teacher, scheduler_teacher, 
						  losses, device, epoch, writer_teacher, total_train_step_teacher, step_teacher,
						  total_fuse_loss_teacher, total_grad_loss_teacher, total_train_loss_teacher):
	"""处理教师网络的一个批次"""
	names, ir_mask, vi_mask, label, ir, y, cb, cr, label_mask = data.values()
	ir_mask, vi_mask, ir, y, cb, cr, label_mask = utils.togpu_7(device, ir_mask, vi_mask, ir, y, cb, cr, label_mask)
	
	# 1. 计算教师网络的输出
	output_teacher, intermediate_outputs_teacher = model_teacher(y, ir, vi_mask, ir_mask)
	fuse_loss_teacher, loss_grad_teacher = model_teacher.loss_cal(output_teacher, y, ir)
	loss_teacher = fuse_loss_teacher + loss_grad_teacher

	# 2. 如果在分割阶段，添加分割损失
	loss_segment = None
	if epoch >= args.ini_epoch: 
		output_colored = utils.YCrCb2RGB(torch.cat((output_teacher, cb, cr), dim=1))
		
		# 使用一个批处理函数来处理分割，减少内存使用
		output_masks = []
		try:
			for j in range(output_colored.size(0)):
				# 分批处理图像以减少内存使用
				single_image = output_colored[j]
				single_image_np = single_image.detach().cpu().permute(1, 2, 0).numpy()
				output_pil = Image.fromarray(np.uint8(single_image_np))
				
				# 使用低内存模式进行分割
				output_mask = segment(output_pil)
				output_masks.append(output_mask)
				
			# 只有在成功处理所有图像时才计算分割损失
			if len(output_masks) == output_colored.size(0):
				output_masks = torch.stack(output_masks)
				label_mask = label_mask.long()
				loss_segment = 0.3*calculate_loss(output_masks, label_mask, device=device)
				loss_teacher = loss_teacher + loss_segment
		except RuntimeError as e:
			if 'out of memory' in str(e):
				# 如果内存不足，尝试清理内存并跳过分割损失
				torch.cuda.empty_cache()
				teacher_logger.warning("Skipping segmentation due to OOM")
			else:
				raise e
	
	# 3. 如果在学生网络训练阶段，添加学生网络的损失
	total_loss_student_new = None
	if epoch >= args.student_epoch:
		try:
			# 使用当前更新的学生网络处理相同的数据
			with torch.no_grad():  
				output_student, intermediate_outputs_student = model_student(y, ir)
			
			# 计算学生网络的损失
			loss_student_new, fuse_loss_student_new, loss_grad_student_new, contrast_loss_student_new, _ = losses.cal(
				output_teacher, y, ir, output_student, vi_mask, ir_mask
			)
			
			# 计算中间层损失
			loss_middle_new = 0.01*calculate_cosine_similarity_loss(intermediate_outputs_teacher, intermediate_outputs_student)
			
			# 添加到教师的总损失中
			total_loss_student_new = 0.1*loss_student_new + loss_middle_new
			loss_teacher += total_loss_student_new
			
			# 清理内存
			del output_student, intermediate_outputs_student
			torch.cuda.empty_cache()
		except RuntimeError as e:
			if 'out of memory' in str(e):
				# 如果内存不足，跳过学生网络的损失
				torch.cuda.empty_cache()
				teacher_logger.warning("Skipping student loss calculation due to OOM")
			else:
				raise e

	# 4. 打印损失信息
	new_lr = scheduler_teacher.get_lr()[0]
	if epoch >= args.ini_epoch and epoch < args.student_epoch and loss_segment is not None:
		print('Teacher:lr_other:', scheduler_teacher.get_lr()[0], 'lr: ', new_lr, 
			  ', fuse_loss: ', fuse_loss_teacher.item(), 'grad_loss: ', loss_grad_teacher.item(), 
			  'loss: ', loss_teacher.item(), "seg_loss:", loss_segment.item())
	elif epoch >= args.student_epoch:
		if loss_segment is not None and total_loss_student_new is not None:
			print('Teacher:lr_other:', scheduler_teacher.get_lr()[0], 'lr: ', new_lr, 
				  ', fuse_loss: ', fuse_loss_teacher.item(), 'grad_loss: ', loss_grad_teacher.item(), 
				  'loss: ', loss_teacher.item(), "seg_loss:", loss_segment.item(),
				  "loss_student_new:", total_loss_student_new.item())
		elif total_loss_student_new is not None:
			print('Teacher:lr_other:', scheduler_teacher.get_lr()[0], 'lr: ', new_lr, 
				  ', fuse_loss: ', fuse_loss_teacher.item(), 'grad_loss: ', loss_grad_teacher.item(), 
				  'loss: ', loss_teacher.item(), "loss_student_new:", total_loss_student_new.item())
		else:
			print('Teacher:lr_other:', scheduler_teacher.get_lr()[0], 'lr: ', new_lr, 
				  ', fuse_loss: ', fuse_loss_teacher.item(), 'grad_loss: ', loss_grad_teacher.item(), 
				  'loss: ', loss_teacher.item())
	else:
		print('Teacher:lr_other:', scheduler_teacher.get_lr()[0], 'lr: ', new_lr, 
			  ', fuse_loss: ', fuse_loss_teacher.item(), 'grad_loss: ', loss_grad_teacher.item(), 
			  'loss: ', loss_teacher.item())

	if not math.isfinite(loss_teacher.item()):
		teacher_logger.info("Loss is {}, stopping training".format(loss_teacher.item()))
		sys.exit(1)

	# 5. 更新教师网络参数
	optimizer_teacher.zero_grad()
	loss_teacher.backward()
	optimizer_teacher.step()
	scheduler_teacher.step()

	# 更新累计损失
	total_fuse_loss_teacher += fuse_loss_teacher.item()
	total_grad_loss_teacher += loss_grad_teacher.item()
	total_train_loss_teacher += loss_teacher.item()

	# 记录和可视化
	if total_train_step_teacher % args.report_freq == 0:
		report_path = EXP_path_Teacher

		output_colored = utils.YCrCb2RGB(torch.cat((output_teacher, cb, cr), dim=1))
		input_vis = utils.YCrCb2RGB(torch.cat((y, cb, cr), dim=1))
		y2rgb = torch.cat((y, y, y), dim=1)

		torchvision.utils.save_image(input_vis, os.path.join(report_path, 'vis.png'))
		torchvision.utils.save_image(y2rgb, os.path.join(report_path, 'y.png'))
		torchvision.utils.save_image(ir, os.path.join(report_path, 'ir.png'))
		torchvision.utils.save_image(output_teacher, os.path.join(report_path, 'output.png'))
		torchvision.utils.save_image(output_colored, os.path.join(color_path_Teacher,  'output_color.png'))
		torchvision.utils.save_image(output_colored, os.path.join(color_path_Teacher,  f'output_colored{total_train_step_teacher}.png'))

		if not os.path.exists(os.path.join(report_path, 'masks')):
			os.makedirs(os.path.join(report_path, 'masks'))                            
		vi_mask_split = torch.split(vi_mask, 1, dim=1)
		for idx2, y_m in enumerate(vi_mask_split):
			torchvision.utils.save_image(y_m, os.path.join(report_path, 'masks', 'vi_{}.png'.format(idx2)))

		ir_mask_split = torch.split(ir_mask, 1, dim=1)
		for idx2, ir_m in enumerate(ir_mask_split):
			torchvision.utils.save_image(ir_m, os.path.join(report_path, 'masks', 'ir_{}.png'.format(idx2)))

		# 根据不同阶段记录不同的日志信息
		if epoch >= args.ini_epoch and epoch < args.student_epoch and loss_segment is not None:
			log_message = "Teacher:step: {}, total_step: {}, lr: {}, fuse_loss: {}, grad loss: {}, seg_loss: {}, loss: {}".format(
				step_teacher, total_train_step_teacher, new_lr, fuse_loss_teacher.item(), loss_grad_teacher.item(), loss_segment.item(), loss_teacher.item())
		elif epoch >= args.student_epoch and loss_segment is not None and total_loss_student_new is not None:
			log_message = "Teacher:step: {}, total_step: {}, lr: {}, fuse_loss: {}, grad loss: {}, seg_loss: {}, loss_student_new: {}, loss: {}".format(
				step_teacher, total_train_step_teacher, new_lr, fuse_loss_teacher.item(), loss_grad_teacher.item(), loss_segment.item(), total_loss_student_new.item(), loss_teacher.item())
		elif epoch >= args.student_epoch and total_loss_student_new is not None:
			log_message = "Teacher:step: {}, total_step: {}, lr: {}, fuse_loss: {}, grad loss: {}, loss_student_new: {}, loss: {}".format(
				step_teacher, total_train_step_teacher, new_lr, fuse_loss_teacher.item(), loss_grad_teacher.item(), total_loss_student_new.item(), loss_teacher.item())
		else:
			log_message = "Teacher:step: {}, total_step: {}, lr: {}, fuse_loss: {}, grad loss: {}, loss: {}".format(
				step_teacher, total_train_step_teacher, new_lr, fuse_loss_teacher.item(), loss_grad_teacher.item(), loss_teacher.item())

		teacher_logger.info(log_message)

		writer_teacher.add_scalar("fuse_loss", fuse_loss_teacher.item(), total_train_step_teacher)
		writer_teacher.add_scalar("grad_loss", loss_grad_teacher.item(), total_train_step_teacher)
		writer_teacher.add_scalar("train_loss", loss_teacher.item(), total_train_step_teacher)
		writer_teacher.add_images("vis", input_vis, total_train_step_teacher)
		writer_teacher.add_images("y", y2rgb, total_train_step_teacher)
		writer_teacher.add_images("ir", ir, total_train_step_teacher)
		writer_teacher.add_images("output_colored", output_colored, total_train_step_teacher)
		print(names)

	# 保存模型检查点
	if total_train_step_teacher % args.test_freq == 0:
		torch.save(model_teacher.state_dict(), os.path.join(model_path_Teacher, '{}.pt'.format(total_train_step_teacher)))
		teacher_logger.info('saving pt {}'.format(total_train_step_teacher))
	
	# 清理内存
	del output_teacher, intermediate_outputs_teacher
	torch.cuda.empty_cache()
	
	return total_fuse_loss_teacher, total_grad_loss_teacher, total_train_loss_teacher

def setup_experiment_directories(args):
    """
    为教师和学生模型设置实验目录结构
    
    参数:
        args: 包含 save 属性的参数对象
    """
    global EXP_path_Teacher, inference_dir_Teacher, model_path_Teacher, result_path_Teacher, result_mask_path, color_path_Teacher
    global EXP_path_Student, inference_dir_Student, model_path_Student, result_path_Student, color_path_Student
    global teacher_logger, student_logger
    
    # 确保基础目录存在
    base_dir = './result'
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    # 确保 Teacher 和 Student 目录存在
    teacher_base = os.path.join(base_dir, 'Teacher')
    student_base = os.path.join(base_dir, 'Student')
    if not os.path.isdir(teacher_base):
        os.makedirs(teacher_base)
    if not os.path.isdir(student_base):
        os.makedirs(student_base)

    # 创建实验目录名称
    args.save = 'T_EXP-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    
    # 创建 Teacher 相关目录
    EXP_path_Teacher = os.path.join(teacher_base, args.save)
    if not os.path.isdir(EXP_path_Teacher):
        os.makedirs(EXP_path_Teacher)
    
    # 创建 Teacher 子目录并设置全局变量
    inference_dir_Teacher = os.path.join(EXP_path_Teacher, 'inference')
    model_path_Teacher = os.path.join(EXP_path_Teacher, 'model')
    result_path_Teacher = os.path.join(EXP_path_Teacher, 'result')
    result_mask_path = os.path.join(EXP_path_Teacher, 'result_mask')
    color_path_Teacher = os.path.join(EXP_path_Teacher, 'color')
    
    teacher_subdirs = [inference_dir_Teacher, model_path_Teacher, result_path_Teacher, result_mask_path, color_path_Teacher]
    
    for dir_path in teacher_subdirs:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
    
    # 创建 Student 相关目录
    EXP_path_Student = os.path.join(student_base, args.save)
    if not os.path.isdir(EXP_path_Student):
        os.makedirs(EXP_path_Student)
    
    # 创建 Student 子目录并设置全局变量
    inference_dir_Student = os.path.join(EXP_path_Student, 'inference')
    model_path_Student = os.path.join(EXP_path_Student, 'model')
    result_path_Student = os.path.join(EXP_path_Student, 'result')
    color_path_Student = os.path.join(EXP_path_Student, 'color')
    
    student_subdirs = [inference_dir_Student, model_path_Student, result_path_Student, color_path_Student]
    
    for dir_path in student_subdirs:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
    
    # 设置日志格式和记录器
    log_format = '%(asctime)s %(message)s'
    
    # 配置 Teacher 日志记录器
    teacher_logger = logging.getLogger('TeacherLogger')
    teacher_logger.setLevel(logging.INFO)
    teacher_logger.handlers = []  # 清除任何现有的处理程序
    
    teacher_stream_handler = logging.StreamHandler(sys.stdout)
    teacher_stream_handler.setFormatter(logging.Formatter(log_format))
    teacher_file_handler = logging.FileHandler(os.path.join(EXP_path_Teacher, 'T_EXP.log'))
    teacher_file_handler.setFormatter(logging.Formatter(log_format))
    
    teacher_logger.addHandler(teacher_stream_handler)
    teacher_logger.addHandler(teacher_file_handler)

    # 配置 Student 日志记录器
    student_logger = logging.getLogger('StudentLogger')
    student_logger.setLevel(logging.INFO)
    student_logger.handlers = []  # 清除任何现有的处理程序
    
    student_stream_handler = logging.StreamHandler(sys.stdout)
    student_stream_handler.setFormatter(logging.Formatter(log_format))
    student_file_handler = logging.FileHandler(os.path.join(EXP_path_Student, 'S_EXP.log'))
    student_file_handler.setFormatter(logging.Formatter(log_format))
    
    student_logger.addHandler(student_stream_handler)
    student_logger.addHandler(student_file_handler)



def calculate_cosine_similarity_loss(intermediate_outputs_student, intermediate_outputs_teacher):
	# 确保两个元组长度相等
	assert len(intermediate_outputs_student) == len(intermediate_outputs_teacher), "Input tuples must have the same length"

	total_loss = 0.0
	for student, teacher in zip(intermediate_outputs_student, intermediate_outputs_teacher):

		cosine_sim = F.cosine_similarity(student.flatten(1), teacher.flatten(1), dim=1)

		cosine_loss = 1 - cosine_sim.mean()
		total_loss += cosine_loss

	return total_loss

if __name__=='__main__':
	main()
















