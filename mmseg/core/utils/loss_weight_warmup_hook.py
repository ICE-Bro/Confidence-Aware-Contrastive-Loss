from mmcv.runner import HOOKS, Hook
import torch.nn as nn

@HOOKS.register_module()
class loss_weight_warmup_hook(Hook):

    def __init__(self, start_iter=5000,
                 loss_name='loss_spcontrast'):
        self.start_iter = start_iter
        self.loss_name = loss_name

    def judge(self, name):
        if isinstance(self.loss_name, list):
            return name in self.loss_name
        else:
            return name == self.loss_name

    def before_run(self, runner):
        cur_iter = runner.iter
        if cur_iter != 0:
            return
        if isinstance(runner.model.module.decode_head, nn.ModuleList):
            for _, decode_head in enumerate(runner.model.module.decode_head):
                if isinstance(decode_head.loss_decode, nn.ModuleList):
                    for _, loss_decode in enumerate(decode_head.loss_decode):
                        name = loss_decode._loss_name
                        if self.judge(name):
                            loss_decode.weight_warmup_start()
                            runner.logger.info("%s is fixed to 0 for warmup, warmup iters is %s ", name,
                                               self.start_iter)
                else:
                    name = decode_head.loss_decode._loss_name
                    if self.judge(name):
                        decode_head.loss_decode.weight_warmup_start()
                        runner.logger.info("%s is fixed to 0 for warmup, warmup iters is %s ", name,
                                           self.start_iter)
        else:
            if isinstance(runner.model.module.decode_head.loss_decode, nn.ModuleList):
                for _, loss_decode in enumerate(runner.model.module.decode_head.loss_decode):
                    name = loss_decode._loss_name
                    if self.judge(name):
                        loss_decode.weight_warmup_start()
                        runner.logger.info("%s is fixed to 0 for warmup, warmup iters is %s ", name,
                                           self.start_iter)
            else:
                name = runner.model.module.decode_head.loss_decode._loss_name
                if self.judge(name):
                    runner.model.module.decode_head.loss_decode.weight_warmup_start()
                    runner.logger.info("%s is fixed to 0 for warmup, warmup iters is %s ", name,
                                       self.start_iter)

        if hasattr(runner.model.module, 'auxiliary_head') is False:
            return

        if isinstance(runner.model.module.auxiliary_head, nn.ModuleList):
            for _, auxiliary_head in enumerate(runner.model.module.auxiliary_head):
                if isinstance(auxiliary_head.loss_decode, nn.ModuleList):
                    for _, loss_decode in enumerate(auxiliary_head.loss_decode):
                        name = loss_decode._loss_name
                        if self.judge(name):
                            loss_decode.weight_warmup_start()
                            runner.logger.info("%s is fixed to 0 for warmup, warmup iters is %s ", name,
                                               self.start_iter)
                else:
                    name = auxiliary_head.loss_decode._loss_name
                    if self.judge(name):
                        auxiliary_head.loss_decode.weight_warmup_start()
                        runner.logger.info("%s is fixed to 0 for warmup, warmup iters is %s ", name,
                                           self.start_iter)
        else:
            if isinstance(runner.model.module.auxiliary_head.loss_decode, nn.ModuleList):
                for _, loss_decode in enumerate(runner.model.module.auxiliary_head.loss_decode):
                    name = loss_decode._loss_name
                    if self.judge(name):
                        loss_decode.weight_warmup_start()
                        runner.logger.info("%s is fixed to 0 for warmup, warmup iters is %s ", name,
                                           self.start_iter)
            else:
                name = runner.model.module.auxiliary_head.loss_decode._loss_name
                if self.judge(name):
                    runner.model.module.auxiliary_head.loss_decode.weight_warmup_start()
                    runner.logger.info("%s is fixed to 0 for warmup, warmup iters is %s ", name,
                                       self.start_iter)



    def after_iter(self, runner):
        cur_iter = runner.iter
        if cur_iter == self.start_iter:
            if isinstance(runner.model.module.decode_head, nn.ModuleList):
                for _, decode_head in enumerate(runner.model.module.decode_head):
                    if isinstance(decode_head.loss_decode, nn.ModuleList):
                        for _, loss_decode in enumerate(decode_head.loss_decode):
                            name = loss_decode._loss_name
                            if self.judge(name):
                                weight = loss_decode.weight_warmup_end()
                                runner.logger.info("The weight of %s is set to %s ", name, weight)
                    else:
                        name = decode_head.loss_decode._loss_name
                        if self.judge(name):
                            weight = decode_head.loss_decode.weight_warmup_end()
                            runner.logger.info("The weight of %s is set to %s ", name, weight)
            else:
                if isinstance(runner.model.module.decode_head.loss_decode, nn.ModuleList):
                    for _, loss_decode in enumerate(runner.model.module.decode_head.loss_decode):
                        name = loss_decode._loss_name
                        if self.judge(name):
                            weight = loss_decode.weight_warmup_end()
                            runner.logger.info("The weight of %s is set to %s ", name, weight)
                else:
                    name = runner.model.module.decode_head.loss_decode._loss_name
                    if self.judge(name):
                        weight = runner.model.module.decode_head.loss_decode.weight_warmup_end()
                        runner.logger.info("The weight of %s is set to %s ", name, weight)

            if hasattr(runner.model.module, 'auxiliary_head') is False:
                return

            if isinstance(runner.model.module.auxiliary_head, nn.ModuleList):
                for _, auxiliary_head in enumerate(runner.model.module.auxiliary_head):
                    if isinstance(auxiliary_head.loss_decode, nn.ModuleList):
                        for _, loss_decode in enumerate(auxiliary_head.loss_decode):
                            name = loss_decode._loss_name
                            if self.judge(name):
                                weight = loss_decode.weight_warmup_end()
                                runner.logger.info("The weight of %s is set to %s ", name, weight)
                    else:
                        name = auxiliary_head.loss_decode._loss_name
                        if self.judge(name):
                            weight = auxiliary_head.loss_decode.weight_warmup_end()
                            runner.logger.info("The weight of %s is set to %s ", name, weight)
            else:
                if isinstance(runner.model.module.auxiliary_head.loss_decode, nn.ModuleList):
                    for _, loss_decode in enumerate(runner.model.module.auxiliary_head.loss_decode):
                        name = loss_decode._loss_name
                        if self.judge(name):
                            weight = loss_decode.weight_warmup_end()
                            runner.logger.info("The weight of %s is set to %s ", name, weight)
                else:
                    name = runner.model.module.auxiliary_head.loss_decode._loss_name
                    if self.judge(name):
                        weight = runner.model.module.auxiliary_head.loss_decode.weight_warmup_end()
                        runner.logger.info("The weight of %s is set to %s ", name, weight)


@HOOKS.register_module()
class loss_weight_final_stop_hook(Hook):

    def __init__(self, start_iter=35000,
                 loss_name='loss_spcontrast'):
        self.start_iter = start_iter
        self.loss_name = loss_name

    def judge(self, name):
        if isinstance(self.loss_name, list):
            return name in self.loss_name
        else:
            return name == self.loss_name

    def after_iter(self, runner):
        cur_iter = runner.iter
        if cur_iter == self.start_iter:
            if isinstance(runner.model.module.decode_head, nn.ModuleList):
                for _, decode_head in enumerate(runner.model.module.decode_head):
                    if isinstance(decode_head.loss_decode, nn.ModuleList):
                        for _, loss_decode in enumerate(decode_head.loss_decode):
                            name = loss_decode._loss_name
                            if self.judge(name):
                                weight = loss_decode.weight_warmup_start()
                                runner.logger.info("The weight of %s is set to 0 for final finetune", name)
                    else:
                        name = decode_head.loss_decode._loss_name
                        if self.judge(name):
                            weight = decode_head.loss_decode.weight_warmup_start()
                            runner.logger.info("The weight of %s is set to 0 for final finetune", name)
            else:
                if isinstance(runner.model.module.decode_head.loss_decode, nn.ModuleList):
                    for _, loss_decode in enumerate(runner.model.module.decode_head.loss_decode):
                        name = loss_decode._loss_name
                        if self.judge(name):
                            weight = loss_decode.weight_warmup_start()
                            runner.logger.info("The weight of %s is set to 0 for final finetune", name)
                else:
                    name = runner.model.module.decode_head.loss_decode._loss_name
                    if self.judge(name):
                        weight = runner.model.module.decode_head.loss_decode.weight_warmup_start()
                        runner.logger.info("The weight of %s is set to 0 for final finetune", name)

            if hasattr(runner.model.module, 'auxiliary_head') is False:
                return

            if isinstance(runner.model.module.auxiliary_head, nn.ModuleList):
                for _, auxiliary_head in enumerate(runner.model.module.auxiliary_head):
                    if isinstance(auxiliary_head.loss_decode, nn.ModuleList):
                        for _, loss_decode in enumerate(auxiliary_head.loss_decode):
                            name = loss_decode._loss_name
                            if self.judge(name):
                                weight = loss_decode.weight_warmup_start()
                                runner.logger.info("The weight of %s is set to 0 for final finetune", name)
                    else:
                        name = auxiliary_head.loss_decode._loss_name
                        if self.judge(name):
                            weight = auxiliary_head.loss_decode.weight_warmup_start()
                            runner.logger.info("The weight of %s is set to 0 for final finetune", name)
            else:
                if isinstance(runner.model.module.auxiliary_head.loss_decode, nn.ModuleList):
                    for _, loss_decode in enumerate(runner.model.module.auxiliary_head.loss_decode):
                        name = loss_decode._loss_name
                        if self.judge(name):
                            weight = loss_decode.weight_warmup_start()
                            runner.logger.info("The weight of %s is set to 0 for final finetune", name)
                else:
                    name = runner.model.module.auxiliary_head.loss_decode._loss_name
                    if self.judge(name):
                        weight = runner.model.module.auxiliary_head.loss_decode.weight_warmup_start()
                        runner.logger.info("The weight of %s is set to 0 for final finetune", name)


@HOOKS.register_module()
class project_head_detach_warmup_hook(Hook):

    def __init__(self, start_iter=5000):
        self.start_iter = start_iter

    def before_run(self, runner):
        cur_iter = runner.iter
        if cur_iter != 0:
            return
        if isinstance(runner.model.module.decode_head, nn.ModuleList):
            for _, decode_head in enumerate(runner.model.module.decode_head):
                if hasattr(decode_head, "project_detach"):
                    decode_head.project_detach = True
                    runner.logger.info("The input of the project branch is detached for warmup, warmup iters is %s ",
                                       self.start_iter)
        else:
            if hasattr(runner.model.module.decode_head, "project_detach"):
                runner.model.module.decode_head.project_detach = True
                runner.logger.info("The input of the project branch is detached for warmup, warmup iters is %s ",
                                   self.start_iter)

        if hasattr(runner.model.module, 'auxiliary_head') is False:
            return

        if isinstance(runner.model.module.auxiliary_head, nn.ModuleList):
            for _, auxiliary_head in enumerate(runner.model.module.auxiliary_head):
                if hasattr(auxiliary_head, "project_detach"):
                    auxiliary_head.project_detach = True
                    runner.logger.info("The input of the project branch is detached for warmup, warmup iters is %s ",
                                       self.start_iter)
        else:
            if hasattr(runner.model.module.auxiliary_head, "project_detach"):
                runner.model.module.auxiliary_head.project_detach = True
                runner.logger.info("The input of the project branch is detached for warmup, warmup iters is %s ",
                                   self.start_iter)

    def after_iter(self, runner):
        cur_iter = runner.iter
        if cur_iter == self.start_iter:
            if isinstance(runner.model.module.decode_head, nn.ModuleList):
                for _, decode_head in enumerate(runner.model.module.decode_head):
                    if hasattr(decode_head, "project_detach"):
                        decode_head.project_detach = False
                        runner.logger.info("end warmup")
            else:
                if hasattr(runner.model.module.decode_head, "project_detach"):
                    runner.model.module.decode_head.project_detach = False
                    runner.logger.info("end warmup")

            if hasattr(runner.model.module, 'auxiliary_head') is False:
                return

            if isinstance(runner.model.module.auxiliary_head, nn.ModuleList):
                for _, auxiliary_head in enumerate(runner.model.module.auxiliary_head):
                    if hasattr(auxiliary_head, "project_detach"):
                        auxiliary_head.project_detach = False
                        runner.logger.info("end warmup")
            else:
                if hasattr(runner.model.module.auxiliary_head, "project_detach"):
                    runner.model.module.auxiliary_head.project_detach = False
                    runner.logger.info("end warmup")
