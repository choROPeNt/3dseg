import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.tensorboard import SummaryWriter

from torch3dseg.utils.utils import get_logger, get_number_of_learnable_parameters, create_optimizer, \
    create_lr_scheduler, get_tensorboard_formatter
from torch3dseg.utils.model import get_model
from torch3dseg.utils.losses import get_loss_criterion
from torch3dseg.utils.metrics import get_evaluation_metrics
from torch3dseg.datasets.utils import get_train_loaders

from torchsummary import summary

from torch3dseg.utils import utils

logger = get_logger('UNet3DTrainer')

def create_trainer(config):
    # Create the model
    model = get_model(config['model'])
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
    else:
        logger.info(f'Using {torch.cuda.device_count()} GPU for training')
    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}'")

    model = model.to(device)
    # summary(model, (1,30,30,30))
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')
    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metrics(config)
    # Create data loaders
    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer = create_optimizer(config['optimizer'], model)

    # Create learning rate adjustment strategy
    lr_scheduler = create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

    trainer_config = config['trainer']
    # Create tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)
    return UNet3DTrainer(model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        loss_criterion=loss_criterion,
                        eval_criterion=eval_criterion,
                        tensorboard_formatter=tensorboard_formatter,
                        device=config['device'],
                        loaders=loaders,
                        resume=resume,
                        pre_trained=pre_trained,
                        **trainer_config)

class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs, max_num_iterations,
                 validate_after_iters=200, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 main_eval_metric="MeanIoU",
                 eval_score_higher_is_better=True,
                 tensorboard_formatter=None, skip_train_validation=False,
                 resume=None, pre_trained=None, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.main_eval_metric = main_eval_metric
        self.eval_score_higher_is_better = eval_score_higher_is_better

        logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = utils.load_checkpoint(resume, self.model, self.optimizer)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            # self.checkpoint_dir = os.path.split(resume)[0]
        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            utils.load_checkpoint(pre_trained, self.model, None)
            # if 'checkpoint_dir' not in kwargs:
                # self.checkpoint_dir = os.path.split(pre_trained)[0]

    def fit(self):
        for _ in range(self.num_epochs, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                logger.info(f"Last Valdiation after Stopping criterion is satisfied.. Finishing with validation...")
                ## final validation step
                # set the model in eval mode
                self.model.eval()
                eval_score, eval_loss = self.validate()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)
                # save checkpoint
                self._save_checkpoint(is_best)
                print(f"RESULT: {eval_loss:>8f} \n")
                return
            
            if not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()
            self.num_epochs += 1

        logger.info(f"Last Valdiation after {self.max_num_epochs}. Finishing with validation...")
        ## final validation step
        # set the model in eval mode
        self.model.eval()
        # evaluate on validation set
        eval_score, eval_loss = self.validate()
        # remember best validation metric
        is_best = self._is_best_eval_score(eval_score)
        # save checkpoint
        self._save_checkpoint(is_best)

        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")
        #print statement esp. for OmniOpt (single line!!)
        print(f"RESULT: {eval_loss:>8f} \n")

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        train_eval_scores = {name: utils.RunningAverage() for name in self.eval_criterion}

        # sets the model in training mode
        self.model.train()

        for t in self.loaders['train']:
            logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                        f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}]')

            
            input, target, weight = self._split_training_batch(t)


            output, loss = self._forward_pass(input, target, weight)

            train_losses.update(loss.item(), self._batch_size(input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    

            if self.num_iterations % self.validate_after_iters == 0:
                # set the model in eval mode
                self.model.eval()
                # evaluate on validation set
                eval_score, _ = self.validate()
                # set the model back to training mode
                self.model.train()

                # adjust learning rate if necessary
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_score)

                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                # compute eval criterion
                if not self.skip_train_validation:
                                        
                    # Update each metric separately
                    for name, metric in self.eval_criterion.items():
                        score = metric(output, target)
                        train_eval_scores[name].update(score.item(), self._batch_size(input))
                    
                    # eval_score = self.eval_criterion(output, target)
                    # train_eval_scores.update(eval_score.item(), self._batch_size(input))

                # log stats, params and images
                lr = self.optimizer.param_groups[0]['lr']
                  # Logging averaged results
                avg_scores_str = ', '.join(f'{k}: {v.avg:.4f}' for k, v in train_eval_scores.items())
                logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {avg_scores_str}. lr-rate {lr}')
                # log to tensorboard
                eval_score_avg_dict = {k: v.avg for k, v in train_eval_scores.items()}
                self._log_stats('train', train_losses.avg, eval_score_avg_dict)
                self._log_lr()
                self._log_params()
                # TODO disabled for faster training and less IO
                # self._log_images(input, target, output, 'train_')

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = {name: utils.RunningAverage() for name in self.eval_criterion}

        with torch.no_grad():
            for i, t in enumerate(self.loaders['val']):
                logger.info(f'Validation iteration {i}')

                input, target, weight = self._split_training_batch(t)

                output, loss = self._forward_pass(input, target, weight)

                val_losses.update(loss.item(), self._batch_size(input))

                # if i % 100 == 0:
                #     self._log_images(input, target, output, 'val_')

                # Update all metrics
                for name, metric in self.eval_criterion.items():
                    score = metric(output, target)
                    val_scores[name].update(score.item(), self._batch_size(input))
                
                # eval_score = self.eval_criterion(output, target)
                # val_scores.update(eval_score.item(), self._batch_size(input))

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break
                
            val_score_avg_dict = {k: v.avg for k, v in val_scores.items()}
            self._log_stats('val', val_losses.avg, val_score_avg_dict)
            avg_scores_str = ', '.join(f'{k}: {v.avg:.4f}' for k, v in val_scores.items())
            logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {avg_scores_str}')
            return val_score_avg_dict, val_losses.avg

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output,logits = self.model(input,True)
        # compute the loss
        if weight is None:
            loss = self.loss_criterion(logits, target)
        else:
            loss = self.loss_criterion(logits, target, weight)

        return output, loss

    def _is_best_eval_score(self, eval_score_dict):
        eval_score = eval_score_dict[self.main_eval_metric] ## TODO adjust to a class attrribute
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")

        utils.save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)


    def _log_stats(self, phase, loss_avg, eval_score_avg_dict):
        # Log the loss
        self.writer.add_scalar(f'{phase}_loss_avg', loss_avg, self.num_iterations)

        # Log each evaluation metric individually
        for metric_name, value in eval_score_avg_dict.items():
            self.writer.add_scalar(f'{phase}_{metric_name}_avg', value, self.num_iterations)


    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, prefix=''):
        if self.model.training:
            if isinstance(self.model, nn.DataParallel):
                net = self.model.module
            else:
                net = self.model

            if net.final_activation is not None:
                prediction = net.final_activation(prediction)

        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    def get_loader(self):
        return self.loaders['train']

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
