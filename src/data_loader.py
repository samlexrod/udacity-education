
import pandas as pd
import warnings
import os

class DataLoader:

    data_path = '../data/'

    def __init__(self):

        self.train = None
        self.test = None
        self.sample_submission = None

        self.help()

    def help(self):
        """Help method to provide information about available methods"""

        checkpoints = self._list_checkpoints()

        print(f"""
        Data Loader initialized with data path: {self.data_path}
            - Use get_train_test_data() to get loaded train and test data
            - Use save_feature_engineered_data() to save feature engineered data
            - Use load_feature_engineered(checkpoint_name="available_checkpoint") to load feature engineered data
              
            - Use set_as_category() to set columns as category

        Checkpoints available: {checkpoints}
        """)

    def _list_checkpoints(self):
        """List all checkpoints"""
        return [folder.split('=')[1] for folder in os.listdir(self.data_path) if "checkpoint" in folder]

    def _load_raw(self):
        """Load raw data"""
        self.train = pd.read_csv(self.data_path + 'train.csv')
        self.test = pd.read_csv(self.data_path + 'test.csv')
        self.sample_submission = pd.read_csv(self.data_path + 'sampleSubmission.csv')

    def load_feature_engineered(self, checkpoint_name):
        """Load feature engineered data"""

        # Setting checkpoint name partition
        checkpoint_name = f"/checkpoint={checkpoint_name}/"

        self.train = pd.read_csv(self.data_path + checkpoint_name + 'train_feature_engineered.csv')
        self.test = pd.read_csv(self.data_path + checkpoint_name + 'test_feature_engineered.csv')

    def get_train_test_data(self):
        """Return raw train and test data"""
        if any([self.train is None, self.test is None]):
            print("Loading raw data...")
            self._load_raw()
            print("Raw data loaded successfully!")

        return self.train, self.test
    
    def save_feature_engineered_data(self, train, test, checkpoint_name):
        """Save feature engineered data"""

        # Setting checkpoint name partition
        checkpoint_name = f"/checkpoint={checkpoint_name}/"
        if not os.path.exists(self.data_path + checkpoint_name):
            os.makedirs(self.data_path + checkpoint_name)

        train.to_csv(self.data_path + checkpoint_name + 'train_feature_engineered.csv', index=False)
        test.to_csv(self.data_path + checkpoint_name + 'test_feature_engineered.csv', index=False)
        print("Feature engineered data saved successfully!")

    def set_as_category(self, columns):
        """Return feature engineered data as category for given columns"""

        if any([self.train is None, self.test is None]):
            self.load_raw()
            msg = "load_raw() method called to load raw data automatically. Use load_feature_engineered() to load feature engineered data checkpoint."

        for col in columns:
            self.train[col] = self.train[col].astype("category")
            self.test[col] = self.test[col].astype("category")

        print(f"Columns {columns} set as category successfully!")