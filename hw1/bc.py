import os
import pickle
import argparse
import collections
import numpy as np
import tensorflow as tf
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt

SPLIT_RATIO = 0.80
VERBOSE = True

def _load_expert_data(data_file, num_examples=None):
    expert_data = pickle.load(open(data_file, "rb"))
    observations = expert_data["observations"]
    actions = np.squeeze(expert_data["actions"], axis=1)
    
    if num_examples is not None:
        observations = observations[:num_examples]
        actions = actions[:num_examples]

    if VERBOSE:
        print("Observations: ", observations.shape)
        print("Actions: ", actions.shape)

    return observations, actions

def _prepare_dataset(dataset, save_dir, shuffle=False):
    num_examples = dataset[0].shape[0]
    indices = np.arange(num_examples)
    if shuffle:
        np.random.shuffle(indices)

    indices_train = indices[:int(SPLIT_RATIO * num_examples)]
    indices_val = indices[int(SPLIT_RATIO * num_examples):]

    train_dataset = [d[indices_train] for d in dataset]
    val_dataset = [d[indices_val] for d in dataset]

    # Normalize 'observations'
    mean = np.mean(train_dataset[0], axis=0)
    var = np.var(train_dataset[0], axis=0)
    train_dataset[0] = (train_dataset[0] - mean)/np.sqrt(var)
    val_dataset[0] = (val_dataset[0] - mean)/np.sqrt(var)

    with open(os.path.join(save_dir, 'normalize.pkl'), 'wb') as f:
        pickle.dump({'mean': mean, 'var': var}, f, pickle.HIGHEST_PROTOCOL)

    return train_dataset, val_dataset

def _prepare_test_data(observation, save_dir):
    values = pickle.load(open(os.path.join(save_dir, 'normalize.pkl'), "rb"))
    mean = values['mean']
    var = values['var']

    return (observation-mean)/np.sqrt(var)

def _minibatches(dataset, batch_size, shuffle=True):
    num_examples = dataset[0].shape[0]
    indices = np.arange(num_examples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for minibatch_start in np.arange(0, len(indices), batch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start+batch_size] 
        yield [d[minibatch_indices] for d in dataset]

class BehavioralCloning():
    def __init__(self, params):
        self.params = params

    def __call__(self, observations=None, actions=None):
        self._build_graph() 
        
        if observations is not None and actions is not None:
            best_val_loss = self._run_graph(observations, actions)

            return best_val_loss
        else:
            self._initialize_model(restore_previous=True)

    def _build_graph(self):
        """
        Architecture:
            observations -> hidden layer -> actions
            
            D = input units, H = hidden units, O = output units
            observations (O) = R^{D}
            input-hidden weights (w1) = R^{DXH}
            hidden-output weights (w2) = R^{HXN}
            actions (A) = R^{N}
        """
        params = self.params

        self.O_placeholder = tf.placeholder(tf.float32, [None, params.input_units])
        self.A_placeholder = tf.placeholder(tf.float32, [None, params.output_units])

        l2_regularizer = tf.contrib.layers.l2_regularizer(self.params.l2_scale)
        w1 = tf.layers.Dense(params.hidden_units, kernel_regularizer=l2_regularizer)
        w2 = tf.layers.Dense(params.output_units, kernel_regularizer=l2_regularizer)

        o1 = w1(self.O_placeholder)
        o2 = tf.nn.tanh(o1)
        self.out = w2(o2)

        reg_losses = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(tf.reduce_sum((self.A_placeholder - self.out)**2, axis=1))
        self.train_op = tf.train.AdamOptimizer(learning_rate=params.lr).minimize(self.loss)

    def _run_graph(self, observations, actions):
        # The best_val_loss is used by DAgger and is the best validation loss on the last
        # iteration
        params = self.params
        
        self._initialize_model(restore_previous=params.restore_previous_model)
        best_val_loss = None
        train_set, val_set = _prepare_dataset([observations, actions], params.save_dir)
        for epoch in range(params.num_epochs):
            if epoch % params.print_every_epoch == 0:
                _VERBOSE = True
            else:
                _VERBOSE = False

            if _VERBOSE:
                print("Epoch ", epoch)

            for i, minibatch in enumerate(_minibatches(train_set, params.batch_size, shuffle=True)):
                feed_dict = self._create_feed_dict(*minibatch)
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict)

                if _VERBOSE and (i % params.print_every_iter == 0):
                    print("Iteration: %03d, Loss: %012.8f" % (i,loss))

            val_loss = []
            for minibatch in _minibatches(val_set, params.batch_size, shuffle=False):
                feed_dict = self._create_feed_dict(*minibatch)
                loss = self.sess.run(self.loss, feed_dict)
                val_loss.append(loss)
            val_loss = float(sum(val_loss)/len(val_loss))

            if _VERBOSE:
                print("Validation Loss: %012.8f" % val_loss)

            if best_val_loss is None or best_val_loss >= val_loss:
                best_val_loss = val_loss
                self.saver.save(self.sess, params.save_dir + "model")

        return best_val_loss

    def get_action(self, observation):
        observation = _prepare_test_data(observation, self.params.save_dir)
        feed_dict = self._create_feed_dict(observation)
        action = self.sess.run(self.out, feed_dict)

        return action

    def _create_feed_dict(self, observations, actions=None):
        feed_dict = {self.O_placeholder: observations}
        if actions is not None:
            feed_dict[self.A_placeholder] = actions
        
        return feed_dict

    def _initialize_model(self, restore_previous=False):
        params = self.params
        self.sess = tf.Session()
        self.saver = tf.train.Saver(save_relative_paths=True)
        if restore_previous:
            restored_model = tf.train.latest_checkpoint(params.save_dir)
            if not restored_model:
                raise Exception('No saved model found!')
            self.saver.restore(self.sess, restored_model)
        else:
            self.sess.run(tf.global_variables_initializer())

def _create_model(input_units, output_units, save_dir, num_epochs=500, batch_size=1000, print_every_epoch=1, print_every_iter=10, 
                  hidden_units=128, lr=0.001, restore_previous_model=False, l2_scale=0.0):
    params = {"num_epochs": num_epochs,
              "batch_size": batch_size,
              "print_every_epoch": print_every_epoch,
              "print_every_iter": print_every_iter,
              "input_units": input_units,
              "hidden_units": hidden_units,
              "output_units": output_units,
              "l2_scale": l2_scale,
              "lr": lr,
              "restore_previous_model": restore_previous_model,
              "save_dir": save_dir}

    Params = collections.namedtuple("Params", params.keys())
    params = Params(*params.values())

    bc_model = BehavioralCloning(params)

    return bc_model

def get_units(envname):
    if envname == "Hopper-v2":
        input_units = 11
        output_units = 3
    elif envname == "Humanoid-v2":
        input_units = 376
        output_units = 17
    elif envname == "Reacher-v2":
        input_units = 11
        output_units = 2
    else:
        raise ValueError("Unknown environment %s" % envname)

    return input_units, output_units

def _run_model(envname, save_dir, hidden_units=128, max_timesteps=None, render=True, num_rollouts=20):
    policy_fn = _create_model(*get_units(envname),
                              save_dir,
                              hidden_units=hidden_units)

    policy_fn()
    with tf.Session():
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn.get_action(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0 and VERBOSE: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        return returns, np.mean(returns), np.std(returns), np.array(observations)

def _train_model(expert_data_file, save_dir, num_examples=None):
    observations, actions = _load_expert_data(args.expert_data_file, num_examples)
    model = _create_model(observations.shape[1],
                          actions.shape[1],
                          save_dir,
                          num_epochs=500,
                          batch_size=100,
                          print_every_epoch=1,
                          print_every_iter=100,
                          hidden_units=128,
                          lr=0.001,
                          restore_previous_model=False) 
    model(observations, actions)

def _dataset_size_vs_reward(expert_data_file, envname, save_dir, max_timesteps=None, render=False, num_rollouts=20):
    means = []
    dataset_sizes = []
    for i in range(100, 100000, 5000):
        print("Running model with %d data size" % i)
        _train_model(expert_data_file, save_dir, num_examples=i)
        tf.reset_default_graph()
        _, mean, _, _ = _run_model(envname, save_dir, max_timesteps, render, num_rollouts)
        tf.reset_default_graph()

        dataset_sizes.append(i)
        means.append(mean)

    plt.title("Dataset Size VS Reward")
    plt.plot(dataset_sizes, means)
    plt.xlabel("Dataset Size")
    plt.ylabel("Reward")
    plt.savefig(save_dir + "/size_vs_reward.png")


########################################################################################################################

class DAgger():
    def __init__(self, envname, expert_policy_file, expert_data_file, save_dir, max_timesteps=None):
        # envname: the environment to train the model on
        # expert_data_file: file that contains initial expert data
        # save_dir: directory to save models at

        self.envname = envname
        self.expert_policy_file = expert_policy_file
        self.expert_data_file = expert_data_file
        self.save_dir = save_dir 
        self.max_timesteps = max_timesteps 

    def __call__(self, num_iterations):
        old_dataset = _load_expert_data(self.expert_data_file, num_examples=1000)
        best_val_loss = None

        for i in range(num_iterations):
            print("DAgger Iteration %d" % i)
            tf.reset_default_graph()
            best_val_loss = self._train(*old_dataset, restore_previous_model=False)
            tf.reset_default_graph()
            observations = self._run_policy()
            tf.reset_default_graph()
            new_dataset = self._run_expert(observations)
            old_dataset = self._aggregate(old_dataset, new_dataset)
        print("Best val loss: ", best_val_loss)

    def _train(self, observations, actions, restore_previous_model):
        print("Training model")
        self.hidden_units = 128
        model = _create_model(observations.shape[1],
                              actions.shape[1],
                              self.save_dir,
                              num_epochs=500,
                              batch_size=100,
                              print_every_epoch=20,
                              print_every_iter=50,
                              hidden_units=self.hidden_units,
                              lr=0.001,
                              restore_previous_model=restore_previous_model,
                              l2_scale=0.0) 

        best_val_loss = model(observations, actions)

        return best_val_loss

    def _run_policy(self):
        print("Running policy to get observations")
        _, mean, std, observations = _run_model(self.envname,
                                           self.save_dir,
                                           hidden_units=self.hidden_units,
                                           max_timesteps=self.max_timesteps,
                                           render=False,
                                           num_rollouts=100)

        print("Mean return: ", mean, "Std. return: ", std)
        return observations

    def _run_expert(self, observations):
        print("Getting expert labels")
        policy_fn = load_policy.load_policy(self.expert_policy_file)
        with tf.Session():
            tf_util.initialize()

            actions = []
            for observation in observations:
                action = policy_fn(observation[None,:])
                actions.append(action)
        actions = np.squeeze(np.array(actions), axis=1)

        return observations, actions
 
    def _aggregate(self, old_dataset, new_dataset):
        print("Aggregating dataset")
        print("New data size: ", new_dataset[0].shape[0] + old_dataset[0].shape[0])
        return [np.concatenate((i,j), axis=0) for i, j in zip(old_dataset, new_dataset)]

def _train_model_using_DAgger(envname,
                              expert_policy_file,
                              expert_data_file,
                              save_dir,
                              num_iterations=10,
                              max_timesteps=None):

    model = DAgger(envname, expert_policy_file, expert_data_file, save_dir, max_timesteps)
    model(num_iterations)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--train_model", action='store_true')
    parser.add_argument("--DAgger", action='store_true')
    parser.add_argument("--experiment_hparams", action='store_true')
    parser.add_argument("--expert_policy_file", type=str)
    parser.add_argument("--expert_data_file", type=str)
    parser.add_argument("--envname", type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    parser.add_argument("--DAgger_iterations", type=int, default=10)
    parser.add_argument("--hidden_units", type=int, default=128)

    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    if args.experiment_hparams:
        VERBOSE = False
        _dataset_size_vs_reward(args.expert_data_file, args.envname, args.save_dir, args.max_timesteps, args.render, args.num_rollouts)
    elif args.train_model:
        if args.DAgger:
            _train_model_using_DAgger(args.envname,
                                      args.expert_policy_file,
                                      args.expert_data_file,
                                      args.save_dir,
                                      args.DAgger_iterations,
                                      args.max_timesteps)
        else:
            _train_model(args.expert_data_file, args.save_dir)
    else:
        returns, mean, var, _ = _run_model(args.envname,
                                           args.save_dir,
                                           hidden_units=args.hidden_units,
                                           max_timesteps=args.max_timesteps,
                                           render=args.render,
                                           num_rollouts=args.num_rollouts)
        print('returns', returns)
        print('mean return', mean)
        print('std of return', var)
