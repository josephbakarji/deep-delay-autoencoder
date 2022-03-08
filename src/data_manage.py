from sklearn.model_selection import train_test_split

class DataStruct:
    # t: time with length len(t)
    # x: embedding matrix with size ((len(t)-n_delays)*n_ics, n_delays)
    # z: original solution with size (n_ics, len(t), 3)
    def __init__(self, name=''):
        self.name = name
        self.t = None
        self.x = None
        self.dx = None
        self.ddx = None
        self.z = None
        self.dz = None
        self.ddz = None
        self.sindy_coefficients = None
        self.modes = None
        self.y_spatial = None

    def training_split(self, train_ratio, shuffle=False):
        self.train = DataStruct('train_set')
        self.test = DataStruct('test_set')
        
        if self.ddx is None:
            self.train.x, self.test.x, self.train.dx, self.test.dx = \
                train_test_split(self.x ,self.dx , train_size=train_ratio, shuffle=shuffle)
        else:
            self.train.x, self.test.x, self.train.dx, self.test.dx, self.train.ddx, self.test.ddx = \
                train_test_split(self.x ,self.dx ,self.ddx, train_size=train_ratio, shuffle=shuffle)

    def get_train_test(self):
        train_data = [self.train.x, self.train.dx]
        test_data = [self.test.x, self.test.dx]
        return train_data, test_data

    def get_times(self, n_ics=1): # Only works for n_ics = 1, fix.
        train_time = self.t[:self.train.x.shape[0]]
        test_time = self.t[self.train.x.shape[0] : self.train.x.shape[0]+self.test.x.shape[0]]
        return train_time, test_time

