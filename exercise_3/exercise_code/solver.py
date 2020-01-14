import numpy as np
import torch


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
        num_iterations = num_epochs * iter_per_epoch
        # best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        #[inputs, labels] = train_loader.dataset
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_corrects = 0
            # T R A I N I N G
            # for iteration in range(iter_per_epoch):
            for iteration, (inputs, labels) in enumerate(train_loader):
                #[inputs, labels] = train_loader.dataset
                #for inputs, labels in train_loader:
                #inputs = inputs.to(device)
                #labels = labels.to(device)
                # foward pass / prediction
                output = model.forward(inputs)
                # loss
                train_loss = self.loss_func(output, labels)
                self.train_loss_history.append(train_loss)
                train_loss.backward()
                # optimize
                optim.step()
                # Maybe print training loss
                if iteration % log_nth == 0:
                    print('[Iteration {}/{}]    TRAIN loss: {:.4f}'.format(iteration, num_iterations, train_loss))

                # statistics
                _, preds = torch.max(output, 1)
                running_loss += train_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Accuracy
            train_epoch_loss = running_loss / len(train_loader.dataset)
            train_epoch_acc = running_corrects.double() / len(train_loader.dataset)
            self.train_acc_history.append(train_epoch_acc)

            # V A L I D A T I O N
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # foward pass / prediction
                output = model.forward(inputs)
                # loss
                val_loss = self.loss_func(output, labels)
                self.val_loss_history.append(val_loss)
                # statistics
                _, preds = torch.max(output, 1)
                running_loss += val_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Accuracy
            val_epoch_loss = running_loss / len(val_loader.dataset)
            val_epoch_acc = running_corrects.double() / len(val_loader.dataset)
                 # deep copy the model
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
         #       best_model_wts = copy.deepcopy(model.state_dict())
                self.val_acc_history.append(val_epoch_acc)

            print('[Epoch {}/{}     TRAIN acc/loss: {:.4f}/{:.4f}]'.format(epoch, num_epochs - 1, train_epoch_acc, train_loss))
            print('[Epoch {}/{}     VAL acc/loss: {:.4f}/{:.4f}]'.format(epoch, num_epochs - 1, val_epoch_acc, val_loss))
            print('-' * 10)
        print('Best Accuracy: {:4f}'.format(best_acc))
        # load best model weights
        #model.load_state_dict(best_model_wts)
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        print('FINISH.')
