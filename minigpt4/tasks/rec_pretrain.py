"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.common.registry import registry
from minigpt4.tasks.base_task import BaseTask
from minigpt4.tasks.rec_base_task import RecBaseTask


@registry.register_task("rec_pretrain")
class RecPretrainTask(RecBaseTask):
    def __init__(self):
        super().__init__()

    # def evaluation(self, model, data_loader, cuda_enabled=True):
    #     pass
