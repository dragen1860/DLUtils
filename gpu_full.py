import  numpy as np
import  torch
import  argparse
import  os, time


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')


args = parser.parse_args()


 



def main():  

    print('Allocate from gpus:', args.gpu)
    # ================================================ 
    data = []
    for gpuid in args.gpu.split(','):

        gpuid = int(gpuid)

        device = torch.device('cuda:%d'%gpuid)
        total, used = os.popen(
            'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
                ).read().split('\n')[gpuid].split(',')
        total = int(total)
        used = int(used)

        print('GPU:%d mem:'%gpuid, total, 'used:', used) 

        try:
            block_mem = (total - used) / 3.5
            # print(block_mem)
            x = torch.rand((int(block_mem), 32*16, 32*16)).to(device) 
        except RuntimeError as err:
            del x
            print(err)
            block_mem = (total - used) / 4
            # print(block_mem)
            x = torch.rand((int(block_mem), 32*16, 32*16)).to(device) 
            
      
        data.append(x)

    result = os.popen(
        'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
        ).read().split('\n')
    print('before running gpu mem:', result) 
    # ================================================

    while True: 
        for x in data:
            # print(x.device, x.shape)
            torch.inverse(x)

        result = os.popen(
            'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
            ).read().split('\n')
        print('updated gpu mem:', result, end='\r') 


if __name__ == '__main__':
    main()
