import time
from models import create_model
import os
def print_log(logger,message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')

def train(opt, dataset):

    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)
    ##logger ##
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger,opt.name)
    logger.close()

    model = create_model(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        running_l1 = 0.0
        batches = 0
        
        #Training step
        opt.phase='train'
        for i, data in enumerate(dataset):
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            # Basic loss logging (L1) if available
            if hasattr(model, 'loss_G_L1') and model.loss_G_L1 is not None:
                try:
                    l1_val = float(model.loss_G_L1.item())
                except Exception:
                    l1_val = 0.0
                running_l1 += l1_val
                batches += 1

        print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
        model.save('latest')


        avg_l1 = (running_l1 / max(1, batches)) if batches > 0 else 0.0
        epoch_time = time.time() - epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %.3f sec \t avg L1: %.6f' %
              (epoch, opt.niter + opt.niter_decay, epoch_time, avg_l1))
        model.update_learning_rate()
