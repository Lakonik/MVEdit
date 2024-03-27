#! /usr/bin/python

# The MIT License (MIT)

# Copyright (c) 2015 Luke Gaynor

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import os
import sys
from pprint import pprint
from distutils.util import strtobool
from imagepacker.imagepacker import pack_images

def guess_realpath(path):
    """Checks for a file in a path, or in a local path"""
    basename = os.path.basename(path)

    # first test the path
    if os.path.isfile(basename): # try locally
        return os.path.realpath(basename)
    elif os.path.isfile(path): # good, it exists
        return os.path.realpath(path)
    else:
        return None # uh oh
    pass

def main():
    parser = argparse.ArgumentParser(description="Naively pokes obj+mtls")
    parser.add_argument("obj", help="path to the .obj file")
    parser.add_argument("-m", "--material", help="path to the .mtl file")
    parser.add_argument("-o","--output", help="output name, used for image and folder")
    parser.add_argument("-a","--add", nargs="+", help="any additional images to pack")

    parser.add_argument('--no-crop', dest='crop', action='store_false', help="do not attempt to crop textures to just what is used")
    parser.add_argument('--no-tile', dest='tile', action='store_false', help="do not attempt to tile textures outside of UV space (must be cropping)")
    parser.add_argument('--no-wrap', dest='wrap', action='store_false', help="don't shift remaining UV verts into [0,1] space")

    parser.set_defaults(crop=True)

    args = parser.parse_args()

    additional = None
    if args.add:
        additional = [os.path.realpath(a) for a in args.add]

    # change to the obj's directory
    wdir = os.path.dirname(os.path.realpath(args.obj))
    os.chdir(wdir)

    # normalise names and paths
    obj_local_path = os.path.basename(args.obj)
    obj_name = os.path.splitext(os.path.basename(args.obj))[0]

    if args.output:
        output_name = args.output
    else: # default to a reasonable name
        output_name = obj_name + "_packed"

    # we read the entire obj both to check for mtl now, and for processing later
    obj_lines = []
    with open(obj_local_path, "r") as obj_file:
        obj_lines = [x.strip() for x in obj_file.readlines()]

    mtl = args.material
    if not mtl or not os.path.isfile(mtl): # got to find it ourselves!
        print("\ninvalid or missing mtl path!")
        for line in obj_lines:
            if line.startswith("mtllib"): # bingo, material in obj
                mtl = guess_realpath(line[7:])
                print("\ttrying path found in obj", mtl)
                break

    if not mtl or not os.path.isfile(mtl):
        print("cannot find mtl file!")
        # sys.exit(1)
        raise ValueError("cannot find mtl file!")

    print("opening material to determine diffuse textures")
    # establish material paths
    texmap = None
    names = []
    dmaps = []

    mtl_lines = []
    with open(mtl, "r") as mtl_file:
        mtl_lines = [x.strip() for x in mtl_file.readlines()]

    if mtl_lines[0].strip != "# Textures packed with a simple packer":
        mtl_lines.insert(0,"# Textures packed with a simple packer")

    new_mtl_lines = []
    outname = output_name+"_full.png"
    for line in mtl_lines:
        if line.startswith("newmtl"):
            name = line[7:]
            # print("\tsaw material name", name)
            if name and name != "None":
                if len(dmaps) != len(names):
                    # print("\tlast material did not have a diffuse, ignoring")
                    names.pop()
                names.append(name)
            else:
                # print("\tignoring 'None' material")
                continue
        elif line.startswith("map_"):
            mtype,m = line.split(" ", 1)
            if mtype.lower() == "map_kd":
                # dmap = guess_realpath(line[7:])
                dmap = guess_realpath(m)
                if not dmap:
                    # raise ValueError("missing a required texture file " + line)
                    print("\tmissing a required texture file " + line)

                # if dmap not in dmaps:
                dmaps.append(dmap)
                # print("\tsaw texture map", dmap)
                line = " ".join([mtype, outname])
                # line = "map_Kd " + outname
            else:
                # print("\tignoring non-diffuse texture map", mtype)
                continue
        elif line.startswith("d "):
            # print("\tignoring transparency value")
            continue

        new_mtl_lines.append(line)

    if len(dmaps) != len(names):
        # print("\tlast material did not have a diffuse, ignoring")
        names.pop()

    # process out a map of diffuse texture file paths
    assert(len(names) == len(dmaps))
    texmap = dict(zip(names, dmaps))
    print("\nmaterial name -> texture map:")
    pprint(texmap)

    # find what part (if not the entirety) of each diffuse that is used
    if args.crop:
        class AABB():
            def __init__(self, min_x=None, min_y=None, max_x=None, max_y=None):
                self.min_x = min_x
                self.min_y = min_y
                self.max_x = max_x
                self.max_y = max_y

                self.to_tile = False

            def add(self, x,y):
                self.min_x = min(self.min_x, x) if self.min_x is not None else x
                self.min_y = min(self.min_y, y) if self.min_y is not None else y
                self.max_x = max(self.max_x, x) if self.max_x is not None else x
                self.max_y = max(self.max_y, y) if self.max_y is not None else y

            def uv_wrap(self):
                return (self.max_x - self.min_x, self.max_y - self.min_y)

            def tiling(self):
                if self.min_x and self.max_x and self.min_y and self.max_y:
                    if self.min_x < 0 or self.min_y < 0 or self.max_x > 1 or self.max_y > 1:
                        return (self.max_x - self.min_x, self.max_y - self.min_y)
                return None

            def __repr__(self):
                return "({},{}) ({},{})".format(
                    self.min_x,
                    self.min_y,
                    self.max_x,
                    self.max_y
                )
        textents = {name: AABB() for name in set(dmaps)}
        # textents = dict(zip(names, AABB)) # hue

        uv_lines = []
        curr_mtl = None
        used_mtl = set()

        for line_idx, line in enumerate(obj_lines):
            if line.startswith("vt"):
                uv_lines.append(line_idx)
            elif line.startswith("usemtl"):
                mtl_name = line[7:]
                curr_mtl = mtl_name
                # print("changed to", curr_mtl)
            elif line.startswith("f"): # face definitions
                for vertex in line[2:].split(): # individual vertex definitions
                    v_def = vertex.split(sep="/")
                    if len(v_def) >= 2 and v_def[1]: # v or v/t or v/vt/vn or v//vn
                        uv_idx = int(v_def[1]) - 1 # uv indexes start from 1
                        uv_line_idx = uv_lines[uv_idx]
                        uv_line = obj_lines[uv_line_idx][3:]
                        uv = [float(uv.strip()) for uv in uv_line.split()]

                        if curr_mtl and curr_mtl in texmap:
                            used_mtl.add(mtl_name)
                            textents[texmap[curr_mtl]].add(uv[0], uv[1])
                        else:
                            print(curr_mtl, "not in texmap")
                        # get uv values at uv_idx
                        # alter them in the original file

        # pprint(textents)
        # pprint(used_mtl)

        if args.tile:
            # loop through UV AABB's, warning when out of range and prompting
            # to see if the user wishes to tile the texture

            for name, extent in textents.items():
                print(name, extent)
                if extent.tiling():
                    h_w, v_w = extent.tiling()
                    if h_w > 1 or v_w > 1:
                        print("\nWARNING: The following texture has coordinates that imply it tiles {}x{} times:\n\t{}".format(round(h_w, 1), round(v_w, 1), name))
                        print("This may be intentional (i.e. tank track textures), or a sign of problematic UV coordinates.")
                        print("Consider only unwrapping/tiling this if you know that it is intentional.")
                        print("(If you are unsure, just hit enter to answer 'No')")
                        to_tile = False
                        try:
                            to_tile = strtobool(input("Do you want to unroll this wrapping by tiling the texture? [y/N]: "))
                        except ValueError as ve:
                            pass
                        extent.to_tile = to_tile

                        if to_tile:
                            print("Marking texture to be tiled.")
                        else:
                            print("Ignoring texture tiled.")

    else:
        textents = None
    # pprint(dmaps)
    # pack and save textures, get info about new coordinate changes
    print("\ncreating packed texture")
    if additional: # additional images
        print("adding additional images: " + ",".join(additional))
        dmaps.extend(additional)
    output_image, uv_changes = pack_images(list(set(dmaps)), extents=textents) # remove duplicates
    # output_image.show()

    uv_lines = []
    curr_mtl = None

    # apply changes to .obj UV's
    print("\napplying UV changes to obj")
    new_obj_lines = []
    for line_idx, line in enumerate(obj_lines):
        if line.startswith("vt"):
            uv_lines.append(line_idx)
            new_obj_lines.append(line)
        elif line.startswith("usemtl"):
            mtl_name = line[7:]
            curr_mtl = mtl_name
            new_obj_lines.append(line)
        elif line.startswith("f"): # face definitions
            for vertex in line[2:].split(): # individual vertex definitions
                v_def = vertex.split(sep="/")
                if len(v_def) >= 2 and v_def[1]: # v or v/t or v/vt/vn or v//vn
                    uv_idx = int(v_def[1]) - 1 # uv indexes start from 1
                    uv_line_idx = uv_lines[uv_idx]
                    uv_line = obj_lines[uv_line_idx][3:]
                    uv = [float(uv.strip()) for uv in uv_line.split()]

                    if curr_mtl and curr_mtl in texmap:
                        changes = uv_changes[texmap[curr_mtl]]
                        uv[0] = uv[0] * changes["aspect"][0] + changes["offset"][0]
                        uv[1] = uv[1] * changes["aspect"][1] + changes["offset"][1]


                        new_obj_lines[uv_line_idx] = "vt {0} {1}".format(
                            uv[0], uv[1]
                            # (uv[0] * changes["aspect"][0] + changes["offset"][0]),
                            # (uv[1] * changes["aspect"][1] + changes["offset"][1])
                        )

                    # get uv values at uv_idx
                    # alter them in the original file
            new_obj_lines.append(line)
        elif line.startswith("mtllib"): # change mtl file name!
            print("\tupdated obj's mtllib to",output_name+".mtl")
            new_obj_lines.append("mtllib " + output_name+".mtl")
        else:
            new_obj_lines.append(line)

    print("writing new obj, mtl and texture files to:")#, output_name+".obj", output_name+".mtl", outname)
    print("\t",os.path.realpath(output_name+".obj"))
    print("\t",os.path.realpath(output_name+".mtl"))
    print("\t",os.path.realpath(outname))

    if not os.path.exists(output_name):
        os.mkdir(output_name)
    os.chdir(output_name)

    with open(output_name+".obj", "w") as new_obj:
        new_obj.write("\n".join(new_obj_lines))
    with open(output_name+".mtl", "w") as new_mtl:
        new_mtl.write("\n".join(new_mtl_lines))
    output_image.save(outname, format="PNG")

    print("\nRemember to convert the final packed texture into a JPEG if you do not need the transparency.")

    # output_image.show()

if __name__ == '__main__':
    import traceback
    # import os
    # try:
    main()
    # except Exception as ex:
    #     print("Uh oh!")
    #     traceback.print_exc()
    #     os.system('pause')
