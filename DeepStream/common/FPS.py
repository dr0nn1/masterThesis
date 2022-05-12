################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import time
start_time=time.time()
frame_count=0
frame_count1=0
lastFPS=0
class GETFPS:
    def __init__(self,stream_id):
        global start_time
        self.start_time=start_time
        self.is_first=True
        global frame_count
        self.frame_count=frame_count
        global frame_count1
        self.frame_count1=frame_count1
        self.stream_id=stream_id
        self.last_frame_time = start_time
        global lastFPS
        self.lastFPS = 0
    def get_fps(self):
        end_time=time.time()
        if(self.is_first):
            self.start_time=end_time
            self.is_first=False
        if(end_time-self.start_time>5):
            print("**********************FPS*****************************************")
            print("Fps of stream",self.stream_id,"is ", float(self.frame_count)/5.0)
            self.frame_count=0
            self.start_time=end_time
        else:
            self.frame_count=self.frame_count+1
    def print_data(self):
        print('frame_count=',self.frame_count)
        print('start_time=',self.start_time)
        
    def fps(self):
        end_time=time.time()
        if(end_time-self.last_frame_time>2):
            current_fps = float(self.frame_count1)/2.0
            self.last_frame_time=end_time
            self.lastFPS = f'{current_fps:.1f}'
            self.frame_count1=0
        else:
            self.frame_count1=self.frame_count1+1
        return self.lastFPS