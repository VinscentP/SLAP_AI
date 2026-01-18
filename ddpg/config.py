#global variables
ACTION_SPACE_SIZE = 12          #12 motors, network needs to output 12 values
REPLAY_BUFFER_SIZE = 50000 
SERVO_MIN = 10                  #arbitrary number
SERVO_MAX = 10                  #arbitrary number 
NUM_EPISODES = 1000   
DISCOUNT = 0.99   
TAU = 0.01                      #used to control how fast the target networks get updated
EPISODES = 1000
MAX_EP_STEPS = 1000
MINIBATCH_SIZE = 64
MIN_REPLAY_BUFFER_SIZE = 5000
ACTION_NOISE_STD = 0.1          #used to control how much robot should explore via actions
