import glob
import pandas as pd
import os
from random import choice, randint, seed

class DataExtraction:
    def __init__(self):
        self.users = 5
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.path = self.current_path + r".\dataset\user_000"
        self.groups = {}

    def extract(self):
        dfs = pd.DataFrame()

        for i in range(self.users):
            local_path = self.path + str(i)
            filenames = glob.glob(local_path + "\*.csv")
            road_area = 1
            for file in filenames:
                df = pd.read_csv(file,index_col=0)
                df['USER'] = pd.Series([i for x in range(len(df.index))], index=df.index)
                df['ROAD_AREA'] = pd.Series([road_area for x in range(len(df.index))], index=df.index)

                if road_area == 1 and i == 1:
                    dfs = df
                else:
                    dfs = pd.concat([dfs,df])

                road_area += 1
        return dfs
    def grouping(self):
        acceleration = ['ACCELERATION', 'ACCELERATION_Y', 'ACCELERATION_Z']
        distance_info = ['DISTANCE', 'DISTANCE_TO_NEXT_INTERSECTION',
                              'DISTANCE_TO_NEXT_STOP_SIGNAL', 'DISTANCE_TO_NEXT_TRAFFIC_LIGHT_SIGNAL',
                              'DISTANCE_TO_NEXT_VEHICLE', 'DISTANCE_TO_NEXT_YIELD_SIGNAL']
        gearbox = ['GEARBOX', 'CLUTCH_PEDAL']
        lane_info = ['LANE', 'LANE_LATERAL_SHIFT_CENTER', 'LANE_LATERAL_SHIFT_LEFT',
                          'LANE_LATERAL_SHIFT_RIGHT', 'LANE_WIDTH', 'FAST_LANE']
        pedals = ['BRAKE_PEDAL', 'CLUTCH_PEDAL']
        road_angle = ['CURVE_RADIUS', 'STEERING_WHEEL', 'ROAD_ANGLE']
        speed = ['SPEED', 'SPEED_Y', 'SPEED_Z', 'SPEED_NEXT_VEHICLE', 'SPEED_LIMIT']
        turn_indicators = ['INDICATORS', 'INDICATORS_ON_INTERSECTION']
        uncategorized = ['HORN', 'HEADING']

        self.groups = {'1': acceleration, '2': distance_info,
                      '3': gearbox, '4': lane_info, '5': pedals,
                        '6': road_angle, '7': speed, '8': turn_indicators, '9': uncategorized}


    def dataset_split(self, drop=None):
        """
                :param drop: 1: Acceleration, 2: distance information, 3: gearbox, 4: lane information, 5: pedals, 6: road angle, 7: speed, 8: turn indicators, 9: uncategorized
                :return: X: training data, Y: label, Z: road condition
                """
        dataset = self.extract()

        self.grouping()

        if drop!=None:
            for key in self.groups[str(drop)]:
                dataset = dataset.drop(key,axis=1)
                # print(dataset)

        keys = dataset.keys()

        X = dataset.iloc[:,0:len(keys)-2].to_numpy()
        Y = dataset.iloc[:,-2].to_numpy()
        Z = dataset.iloc[:,-1].to_numpy()
        return X, Y, Z

    def TripletDataset(self):
        anchor = self.extract()
        pos = pd.DataFrame()
        neg = pd.DataFrame()


        for i in range(anchor.shape[0]):
            loc = anchor.iloc[i,:]

            temp = anchor
            temp = temp.drop(temp.iloc[i,:])

            pos_data = temp.loc[temp['USER'] == loc['USER']]
            neg_data = temp.loc[temp['USER'] == loc['USER']]

            rnd = randint(0,pos_data.shape[0]-1)
            n_rnd = randint(0,neg_data.shape[0]-1,)

            pos = pos.append(pos_data.iloc[rnd,:])
            neg = neg.append(neg_data.iloc[n_rnd,:])
        return anchor, pos, neg


class TripletDataset:
    def __init__(self):
        self.users = 5
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.path = self.current_path + r".\dataset\user_000"
        self.groups = {}

    def extract(self):
        data = {}

        for i in range(1,self.users+1):
            local_path = self.path + str(i)
            filenames = glob.glob(local_path + "\*.csv")
            road_area = 1
            dfs = pd.DataFrame()
            for file in filenames:
                df = pd.read_csv(file, index_col=0)
                df['USER'] = pd.Series([i for x in range(len(df.index))], index=df.index)
                df['ROAD_AREA'] = pd.Series([road_area for x in range(len(df.index))], index=df.index)

                if road_area == 1:
                    dfs = df
                else:
                    dfs = pd.concat([dfs, df])

                road_area += 1

            data[i] = dfs
        return data

    def create_triplet_dataset(self, rnd_seed=0):
        seed = rnd_seed
        data = self.extract()
        A = pd.DataFrame()      # Anchor
        P = pd.DataFrame()      # Positive
        N = pd.DataFrame()      # Negative
        for user in range(1,self.users+1):
            len = data[user].shape[0]
            for i in range(len):
                A = A.append(data[user].iloc[i,:])
                rnd = choice([j for j in range(0,len) if j not in [i]])
                P = P.append(data[user].iloc[rnd,:])

                rnd_user = choice([j for j in range(1,self.users+1) if j not in [user]])
                rnd_neg = randint(0,data[rnd_user].shape[0]-1)
                # print(data[rnd_user].iloc[rnd_neg,:])
                N = N.append(data[rnd_user].iloc[rnd_neg,:])

        A.to_csv(self.current_path + '\\dataset\\triplet_data\\anchor.csv')
        P.to_csv(self.current_path + '\\dataset\\triplet_data\\positive.csv')
        N.to_csv(self.current_path + '\\dataset\\triplet_data\\negative.csv')
        # return A, P, N

    def load_triplet_data(self):
        data = {}

        data['A'] = pd.read_csv(self.current_path + '\\dataset\\triplet_data\\anchor.csv')
        data['P'] = pd.read_csv(self.current_path + '\\dataset\\triplet_data\\positive.csv')
        data['N'] = pd.read_csv(self.current_path + '\\dataset\\triplet_data\\negative.csv')
        return data



TD = TripletDataset()

dict = TD.extract()

TD.create_triplet_dataset()

data = TD.load_triplet_data()