import numpy as np

from traits.api import HasTraits, Instance, Array, \
    on_trait_change
from traitsui.api import View, Item, HGroup, Group

from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MayaviScene, \
                                MlabSceneModel
import pdb
import argparse
import scipy.io as sio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vol_path')
    args = parser.parse_args()
    vol_path = args.vol_path
#    data = np.load(vol_path)
    mat_contents = sio.loadmat(vol_path)
    data = mat_contents['mergedP']
    m = mlab.pipeline.volume(mlab.pipeline.scalar_field(data), vmin=0.01, vmax=0.1)
    m.configure_traits()
