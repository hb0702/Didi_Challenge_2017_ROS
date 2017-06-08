""" Tracklet XML file generation and parsing
From Udacity Didi-challenge code.
Modified: Jaeil Park
"""

from __future__ import print_function
from xml.etree.ElementTree import ElementTree
import numpy as np
import itertools
from warnings import warn

def writeln(f, string, tab_count, tab_as_space=False):
    tab_spaces = 4
    indent_str = " " * tab_spaces * tab_count if tab_as_space else "\t" * tab_count
    f.write(indent_str + string + "\n")


class Tracklet(object):

    def __init__(self, object_type, l, w, h, first_frame=0):
        self.object_type = object_type
        self.h = h
        self.w = w
        self.l = l
        self.first_frame = first_frame
        self.poses = []

    def write_xml(self, f, class_id, tab_level=0):
        writeln(f, '<item class_id="%d" tracking_level="0" version="1">' % class_id, tab_level)
        tab_level += 1
        class_id += 1
        writeln(f, '<objectType>%s</objectType>' % self.object_type, tab_level)
        writeln(f, '<h>%f</h>' % self.h, tab_level)
        writeln(f, '<w>%f</w>' % self.w, tab_level)
        writeln(f, '<l>%f</l>' % self.l, tab_level)
        writeln(f, '<first_frame>%d</first_frame>' % self.first_frame, tab_level)
        writeln(f, '<poses class_id="%d" tracking_level="0" version="0">' % class_id, tab_level)
        class_id += 1
        tab_level += 1
        writeln(f, '<count>%d</count>' % len(self.poses), tab_level)
        writeln(f, '<item_version>2</item_version>', tab_level)
        first_pose = True
        for p in self.poses:
            if first_pose:
                writeln(f, '<item class_id="%d" tracking_level="0" version="2">' % class_id, tab_level)
                first_pose = False
            else:
                writeln(f, '<item>', tab_level)
            tab_level += 1
            class_id += 1
            writeln(f, '<tx>%f</tx>' % p['tx'], tab_level)
            writeln(f, '<ty>%f</ty>' % p['ty'], tab_level)
            writeln(f, '<tz>%f</tz>' % p['tz'], tab_level)
            writeln(f, '<rx>%f</rx>' % p['rx'], tab_level)
            writeln(f, '<ry>%f</ry>' % p['ry'], tab_level)
            writeln(f, '<rz>%f</rz>' % p['rz'], tab_level)
            writeln(f, '<state>1</state>', tab_level)  # INTERP = 1
            writeln(f, '<occlusion>-1</occlusion>', tab_level) # UNSET = -1
            writeln(f, '<occlusion_kf>-1</occlusion_kf>', tab_level)
            writeln(f, '<truncation>-1</truncation>', tab_level) # UNSET = -1
            writeln(f, '<amt_occlusion>0.0</amt_occlusion>', tab_level)
            writeln(f, '<amt_occlusion_kf>-1</amt_occlusion_kf>', tab_level)
            writeln(f, '<amt_border_l>0.0</amt_border_l>', tab_level)
            writeln(f, '<amt_border_r>0.0</amt_border_r>', tab_level)
            writeln(f, '<amt_border_kf>-1</amt_border_kf>', tab_level)
            tab_level -= 1
            writeln(f, '</item>', tab_level)
        tab_level -= 1
        writeln(f, '</poses>', tab_level)
        writeln(f, '<finished>1</finished>', tab_level)
        tab_level -= 1
        writeln(f, '</item>', tab_level)
        return class_id


class TrackletCollection(object):

    def __init__(self):
        self.tracklets = []

    def write_xml(self, filename):
        tab_level = 0
        with open(filename, mode='w') as f:
            writeln(f, r'<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>', tab_level)
            writeln(f, r'<!DOCTYPE boost_serialization>', tab_level)
            writeln(f, r'<boost_serialization signature="serialization::archive" version="9">', tab_level)
            writeln(f, r'<tracklets class_id="0" tracking_level="0" version="0">', tab_level)
            tab_level += 1
            writeln(f, '<count>%d</count>' % len(self.tracklets), tab_level)
            writeln(f, '<item_version>1</item_version> ', tab_level)
            class_id = 1
            for obj in self.tracklets:
                class_id = obj.write_xml(f, class_id, tab_level)
            tab_level -= 1
            writeln(f, '</tracklets>', tab_level)
            writeln(f, '</boost_serialization> ', tab_level)
            f.close()

STATE_UNSET = 0
STATE_INTERP = 1
STATE_LABELED = 2
stateFromText = {'0': STATE_UNSET, '1': STATE_INTERP, '2': STATE_LABELED}

OCC_UNSET = 255  # -1 as uint8
OCC_VISIBLE = 0
OCC_PARTLY = 1
OCC_FULLY = 2
occFromText = {'-1': OCC_UNSET, '0': OCC_VISIBLE, '1': OCC_PARTLY, '2': OCC_FULLY}

TRUNC_UNSET = 255  # -1 as uint8, but in xml files the value '99' is used!
TRUNC_IN_IMAGE = 0
TRUNC_TRUNCATED = 1
TRUNC_OUT_IMAGE = 2
TRUNC_BEHIND_IMAGE = 3
truncFromText = {
    '-1': TRUNC_UNSET,  # FIXME RW: Added this
    '99': TRUNC_UNSET,  # FIXME RW: Original code had this but 99 is supposed be 'behind'???
    '0': TRUNC_IN_IMAGE,
    '1': TRUNC_TRUNCATED,
    '2': TRUNC_OUT_IMAGE,
    '3': TRUNC_BEHIND_IMAGE}


class TrackletGT(object):
    r""" representation an annotated object track
    Tracklets are created in function parseXML and can most conveniently used as follows:
    for trackletObj in parseXML(trackletFile):
      for translation, rotation, state, occlusion, truncation, amtOcclusion, amt_borders, absoluteFrameNumber in trackletObj:
        ... your code here ...
      #end: for all frames
    #end: for all tracklet.
    absoluteFrameNumber is in range [first_frame, first_frame+num_frames[
    amtOcclusion and amt_borders could be None
    You can of course also directly access the fields objType (string), size (len-3 ndarray), first_frame/num_frames (int),
      trans/rots (num_frames x 3 float ndarrays), states/truncs (len-num_frames uint8 ndarrays), occs (num_frames x 2 uint8 ndarray),
      and for some tracklets amt_occs (num_frames x 2 float ndarray) and amt_borders (num_frames x 3 float ndarray). The last two
      can be None if the xml file did not include these fields in poses
    """

    object_type = None
    size = None  # len-3 float array: (height, width, length)
    first_frame = None
    trans = None  # n x 3 float array (x,y,z)
    rots = None  # n x 3 float array (x,y,z)
    states = None  # len-n uint8 array of states
    occs = None  # n x 2 uint8 array  (occlusion, occlusion_kf)
    truncs = None  # len-n uint8 array of truncation
    amt_occs = None  # None or (n x 2) float array  (amt_occlusion, amt_occlusion_kf)
    amt_borders = None  # None (n x 3) float array  (amt_border_l / _r / _kf)
    num_frames = None

    def __init__(self):
        r""" create Tracklet with no info set """
        self.size = np.nan * np.ones(3, dtype=float)

    def __str__(self):
        r""" return human-readable string representation of tracklet object
        called implicitly in
        print trackletObj
        or in
        text = str(trackletObj)
        """
        return '[Tracklet over {0} frames for {1}]'.format(self.num_frames, self.object_type)

    def __iter__(self):
        r""" returns an iterator that yields tuple of all the available data for each frame
        called whenever code iterates over a tracklet object, e.g. in
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amt_borders, absoluteFrameNumber in trackletObj:
          ...do something ...
        or
        trackDataIter = iter(trackletObj)
        """
        if self.amt_occs is None:
            return itertools.izip(
                self.trans, self.rots, self.states, self.occs, self.truncs,
                itertools.repeat(None), itertools.repeat(None),
                range(self.first_frame, self.first_frame + self.num_frames))
        else:
            return itertools.izip(
                self.trans, self.rots, self.states, self.occs, self.truncs,
                self.amt_occs, self.amt_borders, range(self.first_frame, self.first_frame + self.num_frames))

# end: class TrackletGT


def parse_xml(tracklet_file):
    r""" parse tracklet xml file and convert results to list of Tracklet objects
    :param tracklet_file: name of a tracklet xml file
    :returns: list of Tracklet objects read from xml file
    """

    # convert tracklet XML data to a tree structure
    etree = ElementTree()
    print('Parsing Tracklet file', tracklet_file)
    with open(tracklet_file) as f:
        etree.parse(f)

    # now convert output to list of Tracklet objects
    tracklets_elem = etree.find('tracklets')
    tracklets = []
    tracklet_idx = 0
    num_tracklets = None
    for tracklet_elem in tracklets_elem:
        if tracklet_elem.tag == 'count':
            num_tracklets = int(tracklet_elem.text)
            print('File contains', num_tracklets, 'Tracklets')
        elif tracklet_elem.tag == 'item_version':
            pass
        elif tracklet_elem.tag == 'item':
            new_track = TrackletGT()
            is_finished = False
            has_amt = False
            frame_idx = None
            for info in tracklet_elem:
                if is_finished:
                    raise ValueError('More info on element after finished!')
                if info.tag == 'objectType':
                    new_track.object_type = info.text
                elif info.tag == 'h':
                    new_track.size[0] = float(info.text)
                elif info.tag == 'w':
                    new_track.size[1] = float(info.text)
                elif info.tag == 'l':
                    new_track.size[2] = float(info.text)
                elif info.tag == 'first_frame':
                    new_track.first_frame = int(info.text)
                elif info.tag == 'poses':
                    # this info is the possibly long list of poses
                    for pose in info:
                        if pose.tag == 'count':  # this should come before the others
                            if new_track.num_frames is not None:
                                raise ValueError('There are several pose lists for a single track!')
                            elif frame_idx is not None:
                                raise ValueError('?!')
                            new_track.num_frames = int(pose.text)
                            new_track.trans = np.nan * np.ones((new_track.num_frames, 3), dtype=float)
                            new_track.rots = np.nan * np.ones((new_track.num_frames, 3), dtype=float)
                            new_track.states = np.nan * np.ones(new_track.num_frames, dtype='uint8')
                            new_track.occs = np.nan * np.ones((new_track.num_frames, 2), dtype='uint8')
                            new_track.truncs = np.nan * np.ones(new_track.num_frames, dtype='uint8')
                            new_track.amt_occs = np.nan * np.ones((new_track.num_frames, 2), dtype=float)
                            new_track.amt_borders = np.nan * np.ones((new_track.num_frames, 3), dtype=float)
                            frame_idx = 0
                        elif pose.tag == 'item_version':
                            pass
                        elif pose.tag == 'item':
                            # pose in one frame
                            if frame_idx is None:
                                raise ValueError('Pose item came before number of poses!')
                            for poseInfo in pose:
                                if poseInfo.tag == 'tx':
                                    new_track.trans[frame_idx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ty':
                                    new_track.trans[frame_idx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'tz':
                                    new_track.trans[frame_idx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'rx':
                                    new_track.rots[frame_idx, 0] = float(poseInfo.text)
                                elif poseInfo.tag == 'ry':
                                    new_track.rots[frame_idx, 1] = float(poseInfo.text)
                                elif poseInfo.tag == 'rz':
                                    new_track.rots[frame_idx, 2] = float(poseInfo.text)
                                elif poseInfo.tag == 'state':
                                    new_track.states[frame_idx] = stateFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion':
                                    new_track.occs[frame_idx, 0] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion_kf':
                                    new_track.occs[frame_idx, 1] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'truncation':
                                    new_track.truncs[frame_idx] = truncFromText[poseInfo.text]
                                elif poseInfo.tag == 'amt_occlusion':
                                    new_track.amt_occs[frame_idx, 0] = float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_occlusion_kf':
                                    new_track.amt_occs[frame_idx, 1] = float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_border_l':
                                    new_track.amt_borders[frame_idx, 0] = float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_border_r':
                                    new_track.amt_borders[frame_idx, 1] = float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_border_kf':
                                    new_track.amt_borders[frame_idx, 2] = float(poseInfo.text)
                                    has_amt = True
                                else:
                                    raise ValueError('Unexpected tag in poses item: {0}!'.format(poseInfo.tag))
                            frame_idx += 1
                        else:
                            raise ValueError('Unexpected pose info: {0}!'.format(pose.tag))
                elif info.tag == 'finished':
                    is_finished = True
                else:
                    raise ValueError('Unexpected tag in tracklets: {0}!'.format(info.tag))
            # end: for all fields in current tracklet

            # some final consistency checks on new tracklet
            if not is_finished:
                warn('Tracklet {0} was not finished!'.format(tracklet_idx))
            if new_track.num_frames is None:
                warn('Tracklet {0} contains no information!'.format(tracklet_idx))
            elif frame_idx != new_track.num_frames:
                warn('Tracklet {0} is supposed to have {1} frames, but parser found {1}!'.format(
                    tracklet_idx, new_track.num_frames, frame_idx))
            if np.abs(new_track.rots[:, :2]).sum() > 1e-16:
                warn('Track contains rotation other than yaw!')

            # if amt_occs / amt_borders are not set, set them to None
            if not has_amt:
                new_track.amt_occs = None
                new_track.amt_borders = None

            # add new tracklet to list
            tracklets.append(new_track)
            tracklet_idx += 1

        else:
            raise ValueError('Unexpected tracklet info')
    # end: for tracklet list items

    print('Loaded', tracklet_idx, 'Tracklets')

    # final consistency check
    if tracklet_idx != num_tracklets:
        warn('According to xml information the file has {0} tracklets, but parser found {1}!'.format(
            num_tracklets, tracklet_idx))
    return tracklets
