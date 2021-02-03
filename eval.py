from models.rl_process import rl_eval, seq_eval
from itertools import count
from mail import send_mail

if __name__ == '__main__':
    # for t in count():
    #     print("{} Turn Eval".format(t))
    #     flg = seq_eval(10)
    #     if flg:
    #         print("Break down at {}".format(flg))
    #     else:
    #         break
    rl_eval(10)
    # send_mail("Evaluate Done.")
