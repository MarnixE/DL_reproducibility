import glob
import pandas as pd

class DataExtraction:
    def __init__(self):
        self.users = 5
        self.path = r".\dataset\user_000"
        self.acceleration = ['ACCELERATION','ACCELERATION_Y','ACCELERATION_Z']
        self.distance_info = ['DISTANCE', 'DISTANCE_TO_NEXT_INTERSECTION',
       'DISTANCE_TO_NEXT_STOP_SIGNAL', 'DISTANCE_TO_NEXT_TRAFFIC_LIGHT_SIGNAL',
       'DISTANCE_TO_NEXT_VEHICLE', 'DISTANCE_TO_NEXT_YIELD_SIGNAL']
        self.gearbox = ['GEARBOX','CLUTCH_PEDAL']
        self.lane_info = ['LANE', 'LANE_LATERAL_SHIFT_CENTER', 'LANE_LATERAL_SHIFT_LEFT',
       'LANE_LATERAL_SHIFT_RIGHT', 'LANE_WIDTH', 'FAST_LANE']
        self.pedals = ['BRAKE_PEDAL', 'CLUTCH_PEDAL']
        self.road_angle = ['CURVE_RADIUS','STEERING_WHEEL','ROAD_ANGLE']
        self.speed = ['SPEED','SPEED_Y','SPEED_Z','SPEED_NEXT_VEHICLE','SPEED_LIMIT']
        self.turn_indicators = ['INDICATORS','INDICATORS_ON_INTERSECTION']
        self.uncategorized = ['HORN','HEADING']
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

    # def groupping(self):
    #     data = self.extract()
    #     for i in range(4):


    # def split(self):


DE = DataExtraction()

dict = DE.extract()
