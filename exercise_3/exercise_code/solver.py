import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Running on:', device)
        model.to(device)

        print('START TRAIN.')
        #######################################################################
        # TODO:                                                               #
        # Write your own personal training method for our solver. In each     #
        # epoch iter_per_epoch shuffled training batches are processed. The   #
        # loss for each batch is stored in self.train_loss_history. Every     #
        # log_nth iteration the loss is logged. After one epoch the training  #
        # accuracy of the last mini batch is logged and stored in             #
        # self.train_acc_history. We validate at the end of each epoch, log   #
        # the result and store the accuracy of the entire validation set in   #
        # self.val_acc_history.                                               #
        #                                                                     #
        # Your logging could like something like:                             #
        #   ...                                                               #
        #   [Iteration 700/4800] TRAIN loss: 1.452                            #
        #   [Iteration 800/4800] TRAIN loss: 1.409                            #
        #   [Iteration 900/4800] TRAIN loss: 1.374                            #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                           #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                           #
        #   ...                                                               #
        #######################################################################
        writer = SummaryWriter()
        num_iterations = num_epochs * iter_per_epoch
        best_val_acc = 0.0
        for epoch in range(num_epochs):

            # T R A I N I N G
            for iteration, (inputs, labels) in enumerate(train_loader, 1):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optim.zero_grad()
                # with torch.set_grad_enabled(True):
                # foward pass / prediction
                output = model(inputs)
                # print('Out Shape:', output.shape)
                # loss
                loss = self.loss_func(output, labels)
                loss.backward()
                # optimize
                optim.step()

                self.train_loss_history.append(loss.data.cpu().numpy())
                writer.add_scalar('Train Loss', self.train_loss_history[-1], iteration)
                # t = epoch * iter_per_epoch + iteration
                # Maybe print training loss
                # if t % log_nth == 0:
                if log_nth and iteration % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration {}/{}]    TRAIN loss: {:.4f}'.format\
                        (iteration + epoch * iter_per_epoch, 
                         iter_per_epoch * num_epochs, 
                         train_loss))

            # Accuracy per minibatch
            _, preds = torch.max(output, 1)
            targets_mask = labels>= 0
            train_acc = np.mean((preds == labels)[targets_mask].data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch {}/{}] TRAIN acc/loss: {:.4f}/{:.4f}'.format(epoch + 1,
                                                                       num_epochs,
                                                                       train_acc,
                                                                       train_loss))

            # V A L I D A T I O N
            val_losses = []
            val_scores = []
            model.eval()
            # with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # foward pass / prediction
                output = model.forward(inputs)
                # loss
                loss = self.loss_func(output, labels)
                val_losses.append(val_loss.data.cpu().numpy())

                # Accuracy
                _, pred = torch.max(output, 1)
                targets_mask = targets >= 0
                scores = np.mean((pred == labels)[targets_mask].data.cpu().numpy())
                val_accs.append(scores)


            model.train()
            val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            writer.add_scalar('Validation Loss', val_loss, iteration)

            if log_nth:
                print('[Epoch {}/{}]     VAL acc/loss: {:.4f}/{:.4f}'.format(epoch + 1, 
                                                                             num_epochs, 
                                                                             val_acc, 
                                                                             val_loss))
                print('-' * 30)

            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            # print('[Epoch {}/{}]     TRAIN acc/loss: {:.4f}/{:.4f}'.format(epoch + 1, num_epochs, self.train_acc_history[-1], self.train_loss_history[-1]))
            # print('[Epoch {}/{}]     VAL acc/loss: {:.4f}/{:.4f}'.format(epoch + 1, num_epochs, val_acc, self.val_loss_history[-1]))
            # print('-' * 30)
        # print('Best Accuracy: {:4f}'.format(best_val_acc))
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        print('FINISH.')
