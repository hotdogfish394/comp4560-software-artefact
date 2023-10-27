from config import *
import pandas as pd
from torch.utils.data import DataLoader
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning.pytorch as pl
import os
import sklearn.metrics as metrics
from lightning.pytorch import Trainer

class ModelWrapper:
    def __init__(self, 
                 model_class, 
                 dataset_headings, 
                 dataset, 
                 variables, 
                 transform=None, 
                 model_name=None, 
                 model_params=None,
                 checkpoint_path=None):
        
        # model parameters
        self.model_class = model_class
        self.model = None
        self.model_params = model_params
        self.name = None
        self.last_checkpoint = None

        # intialise model
        if model_params is not None:
            self.model = model_class(**model_params)
        else:
            self.model = model_class()
        
        # rename model
        if model_name is not None:
            self.model.name = model_name

        # name depends on dataset_headings and transform    
        if transform is None:
            self.name = self.model.name + '_' + dataset_headings + '_' + str(SEED)
        else:
            self.name = self.model.name + '_' + dataset_headings + '_'  + transform.__name__ + '_' + str(SEED)
        
        # dataset parameters
        self.dataset_headings = dataset_headings
        self.dataset = dataset
        self.variables = variables
        self.transform = transform
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.load_data()

        # lightning pytorch parameters
        self.logger = TensorBoardLogger(LOG_DIR, name=self.name)
        self.trainer = None
        
        self.early_stop_callback = EarlyStopping(monitor="val_acc", 
                                                 patience=10, 
                                                 mode="max",
                                                 min_delta=0.05,
                                                 check_finite=True)
        
        self.save_dir = os.path.join(os.getcwd(), 'lightning_logs')
        
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(self.save_dir, self.name), 
                                                                monitor="val_acc",
                                                                save_weights_only=True, 
                                                                save_top_k=1, 
                                                                mode="max",
                                                                auto_insert_metric_name = True)

        if checkpoint_path:
            self.last_checkpoint = checkpoint_path
            self.load_checkpoint()

    def load_checkpoint(self):
        if self.last_checkpoint is not None:
            if self.model_params is not None:
                self.model = self.model_class.load_from_checkpoint(self.last_checkpoint,**self.model_params)
            else:
                self.model = self.model_class.load_from_checkpoint(self.last_checkpoint)
            print("Loaded checkpoint: " + self.last_checkpoint)
        else:
            print("No checkpoint found")
        
    def load_data(self):
        """
        Load and preprocess data for training, validation, and testing.

        Returns:
            None
        """
        train_df = pd.read_csv(DATASET_DIR + "/" + self.dataset_headings + '_train.csv')
        train_df = train_df[self.variables]
       
        val_df = pd.read_csv(DATASET_DIR + "/" + self.dataset_headings + '_val.csv')
        val_df = val_df[self.variables]
        
        test_df = pd.read_csv(DATASET_DIR + "/" + self.dataset_headings + '_test.csv')
        test_df = test_df[self.variables]

        # set catid to str
        train_df['CATID'] = train_df['CATID'].astype(str)
        val_df['CATID'] = val_df['CATID'].astype(str)
        test_df['CATID'] = test_df['CATID'].astype(str)

        # drop nans just in case
        train_df = train_df.dropna()
        val_df = val_df.dropna()
        test_df = test_df.dropna()

        # intialise dataset with transform
        train_dataset = self.dataset(train_df, data_transform=self.transform)
        val_dataset = self.dataset(val_df, data_transform=self.transform)
        test_dataset = self.dataset(test_df, data_transform=self.transform)

        # assert no nan values
        has_nan_values = train_df.isnull().values.any() or val_df.isnull().values.any() or test_df.isnull().values.any()
        assert not has_nan_values, "Dataset contains NaN values"
        
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    def train_model(self):
        """
        Train the machine learning model.

        Returns:
            None
        """
        has_all_loaders = self.train_loader is not None and self.val_loader is not None and self.test_loader is not None
        assert has_all_loaders, "Loaders not initialised"
        my_callbacks = [self.early_stop_callback,self.checkpoint_callback]
        self.trainer = pl.Trainer(callbacks=my_callbacks,
                                  min_epochs=10, 
                                  max_epochs=MAX_EPOCHS, 
                                  logger=self.logger)
        
        self.trainer.fit(self.model, self.train_loader, self.val_loader)
        self.last_checkpoint = self.trainer.checkpoint_callback.best_model_path

    def test_model(self):
        """
        Test the machine learning model.

        Returns:
            loss (float): Test loss value.
        """
        trainer = pl.Trainer(logger=False)
        loss = trainer.test(self.model, self.test_loader)
        return loss
    
    def predict_model(self):
        """
        Generate predictions using the trained model.

        Returns:
            (y, preds, cat_ids) (tuple): Predicted labels (y), model predictions (preds), and category IDs (cat_ids).
        """
        trainer = pl.Trainer(logger=False)
        predictions = trainer.predict(self.model, dataloaders=self.test_loader)
        y = []
        preds = []
        cat_ids = []
        for i in range(len(predictions)):
            y_i,preds_i,cat_id_i = predictions[i]
            y.extend(y_i)
            preds.extend(preds_i)
            cat_ids.extend(cat_id_i)
        return (y,preds,cat_ids)
    
    def write_results_classification(self, predictions, test_results):
        """
        Write classification results to a CSV file.

        Args:
            predictions (tuple): Predicted labels (y), model predictions (preds), and category IDs (cat_ids).
            test_results (list): Test results, including accuracy, F1 scores, AUC, and loss.

        Returns:
            None
        """
        y,preds,_ = predictions
        test_acc_balanced = test_results[0]['test_acc']
        test_acc = metrics.accuracy_score(y, preds)
        test_f1_macro = metrics.f1_score(y, preds, average='macro')
        test_f1_micro = metrics.f1_score(y, preds, average='micro')
        test_f1_weighted = metrics.f1_score(y, preds, average='weighted')
        test_auc = metrics.roc_auc_score(y, preds)
        test_loss = test_results[0]['test_loss']

        # append to csv
        df = pd.DataFrame([[self.name, test_acc_balanced, test_acc, test_f1_macro, test_f1_micro, test_f1_weighted, test_auc, test_loss, self.last_checkpoint]], columns=['name', 'test_acc_balanced', 'test_acc', 'test_f1_macro', 'test_f1_micro', 'test_f1_weighted', 'test_auc', 'test_loss', 'checkpoint_path'])
        if not os.path.isfile('classification_results.csv'):
            df.to_csv('classification_results.csv', index=False)
        else:
            df.to_csv('classification_results.csv', mode='a', header=False, index=False)