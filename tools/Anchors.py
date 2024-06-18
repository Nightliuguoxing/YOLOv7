# =================================
# File : Anchors
# Author : LGX
# Description : 
# CREATE TIME : 2023/10/17 9:07
# =================================
# -*- coding: utf-8 -*-
import utils.autoanchor as autoAC

new_anchors = autoAC.kmean_anchors('../data/forest.yaml', 12, 640, 4.0, 1000, True)
print(new_anchors)