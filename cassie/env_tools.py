import os, pathlib

def env_by_name(args):
    file_path = pathlib.Path( os.path.realpath(__file__) )
    env_name = args.env

    if env_name == 'Cassie-v000':
        from .cassiemujoco.cassie import CassieEnv

        models_path = file_path.parent.__str__() + '/cassiemujoco/cassiemujoco/'
        model_file = 'cassie.xml'
        env = CassieEnv(model_path = models_path + model_file)

    elif env_name == 'Cassie-v001':
        from .cassiemujoco.cassie import CassieWalkingEnv

        models_path = file_path.parent.__str__() + '/cassiemujoco/cassiemujoco/'
        model_file = 'cassie.xml'
        data_file = file_path.parent.__str__() + '/trajectory/stepdata.bin'
        env = CassieWalkingEnv(model_path = models_path + model_file, simrate=args.simrate, trajdata_path=data_file)

    elif env_name == 'Cassie-v100':
        from .mujocosim.envs import CassieEnv

        models_path = file_path.parent.__str__() + '/mujocosim/model/'
        model_file = 'cassie.xml'
        env = CassieEnv(model_path = models_path + model_file)

    elif env_name == 'Cassie-v101':
        from .mujocosim.envs import CassieWalkingEnv

        models_path = file_path.parent.__str__() + '/mujocosim/model/'
        model_file = 'cassie.xml'
        data_file = file_path.parent.__str__() + '/trajectory/stepdata.bin'
        env = CassieWalkingEnv(model_path = models_path + model_file, simrate=args.simrate, trajdata_path=data_file)
    else:
        print("Can't find env " + env_name)
    
    return env



