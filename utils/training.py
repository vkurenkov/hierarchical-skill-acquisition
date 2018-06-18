import os
import glob
import h5py
import torch

class Session:
    '''
    Keeps information about training or test session.
    Use switch_group to initialize.
    '''
    def __init__(self, name, testing=False):
        '''
        Loads an existing session or creates a new one.

        name - a string (must not contain '-' symbol)
        testing - whether you test or train (defaults to training)
        '''
        if(not testing):
            dir_path = os.path.join(".", "training")
        else:
            dir_path = os.path.join(".", "testing")
        if(not os.path.isdir(dir_path)):
            os.mkdir(dir_path)

        self.session_dir = os.path.join(dir_path, "session-" + name)
        self.checkpoints_dir = os.path.join(self.session_dir, "checkpoints")
        if(os.path.isdir(self.session_dir)):
            self._open(self.session_dir)
        else:
            self._create(self.session_dir)

    def _create(self, session_dir):
        os.mkdir(session_dir)
        os.mkdir(os.path.join(session_dir, "checkpoints"))

        self._init_log(session_dir)
        self._init_data(session_dir)
    def _open(self, session_dir):
        self._init_log(session_dir)
        self._init_data(session_dir)

    def _init_log(self, session_dir):
        self._logs = open(os.path.join(session_dir, "logs"), "a")
    def _init_data(self, session_dir):
        self.data = h5py.File(os.path.join(session_dir, "data.hdf5"), "a")

    def _create_new_group(self, group_name):
        self.cur_group = self.data.create_group(group_name)
        self.cur_group.create_dataset("reward", shape=(0, 1), maxshape=(None, 1))
        self.cur_group.create_dataset("value_loss", shape=(0, 1), maxshape=(None, 1))
        self.cur_group.create_dataset("a2c_loss", shape=(0, 1), maxshape=(None, 1))
        self.cur_group.create_dataset("total_loss", shape=(0, 1), maxshape=(None, 1))
        self.cur_group.create_dataset("actions_entropy", shape=(0, 1), maxshape=(None, 1))
        self.cur_group.create_dataset("timesteps", shape=(0, 1), maxshape=(None, 1), dtype=int)
    def _append(self, value, dataset):
        size = dataset.size
        dataset.resize(size + 1, 0)
        dataset[size, 0] = value
    def _get_dataset(self, dataset_name, group_name=None):
        if group_name != None:
            return self.data[str(group_name)][dataset_name]
        else:
            return self.cur_group[dataset_name]
            
    def log(self, data):
        '''
        input:
            data - any data
        '''
        self._logs.write(data)
    def log_flush(self):
        self._logs.flush()

    def reward(self, reward):
        self._append(reward, self.cur_group["reward"])
    
    def value_loss(self, loss):
        self._append(loss, self.cur_group["value_loss"])
    
    def a2c_loss(self, loss):
        self._append(loss, self.cur_group["a2c_loss"])
    
    def total_loss(self, loss):
        self._append(loss, self.cur_group["total_loss"])
    
    def actions_entropy(self, entropy):
        self._append(entropy, self.cur_group["actions_entropy"])
    
    def timesteps(self, timesteps):
        self._append(timesteps, self.cur_group["timesteps"])

    def checkpoint_model(self, model, name):
        '''
        input:
            model - torch module
            name - a string (must not contain the '-' sign)
        '''
        checkpoint_files = glob.glob(os.path.join(self.checkpoints_dir, name + "-*"))
        checkpoint_file = os.path.join(self.checkpoints_dir, name + "-")
        if(len(checkpoint_files) == 0):
            checkpoint_file += "0"
        else:
            maximum_id = max([int(f.split("-")[2]) for f in checkpoint_files ])
            checkpoint_file += str(maximum_id + 1)

        torch.save(model, checkpoint_file + ".pt")

    def get_rewards(self, group_id=None):
        return self._get_dataset("reward", group_id)
    def get_value_losses(self, group_id=None):
        return self._get_dataset("value_loss", group_id)
    def get_a2c_losses(self, group_id=None):
        return self._get_dataset("a2c_loss", group_id)
    def get_total_losses(self, group_id=None):
        return self._get_dataset("total_loss", group_id)
    def get_actions_entropies(self, group_id=None):
        return self._get_dataset("actions_entropy", group_id)
    def get_timesteps(self, group_id=None):
        return self._get_dataset("timesteps", group_id)

    def switch_group(self, group_id=None):
        '''
        input:
            group_id - id of the group you want to work on (appends and gets)
                         if None - it switches to new group
        '''
        if(group_id != None):
            self.cur_group = self.data[str(group_id)]
        else:
            if(len(self.data.keys()) > 0):
                max_group_id = max([int(key) for key in self.data.keys()])
                self._create_new_group(str(max_group_id + 1))
            else:
                self._create_new_group("0")

    def close(self):
        self._logs.close()
        self.data.close()

if __name__ == "__main__":
    sess = Session(name="another_terminal")
    sess.switch_group()
    print("Session: Creation is verified.")

    sess.reward(10)
    sess.value_loss(20)
    sess.a2c_loss(30)
    sess.total_loss(40)
    sess.actions_entropy(0.5)
    sess.timesteps(5)

    assert(sess.get_rewards()[0, 0] == 10)
    assert(sess.get_value_losses()[0, 0] == 20)
    assert(sess.get_a2c_losses()[0, 0] == 30)
    assert(sess.get_total_losses()[0, 0] == 40)
    assert(sess.get_actions_entropies()[0, 0] == 0.5)
    assert(sess.get_timesteps()[0, 0] == 5)
    print("Session: Appends are verified.")
    sess.close()

    sess1 = Session(name="another_terminal")
    sess1.switch_group(0)
    assert(sess1.get_rewards()[0, 0] == 10)
    assert(sess1.get_value_losses()[0, 0] == 20)
    assert(sess1.get_a2c_losses()[0, 0] == 30)
    assert(sess1.get_total_losses()[0, 0] == 40)
    assert(sess1.get_actions_entropies()[0, 0] == 0.5)
    assert(sess1.get_timesteps()[0, 0] == 5)
    print("Session: Open existing session is verified.")