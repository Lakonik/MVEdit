# Image Packer
Takes a Wavefront OBJ with many textures and squishes them into a single texture file.

## Why?
I put this together for the purpose of packing complex models with multiple textures (practically any non-trivial model) into a single `.obj` and texture file for use as custom models in [Tabletop Simulator](http://berserk-games.com/tabletop-simulator/).
Initially, I had to do this by hand -- first, combining the textures in GIMP and then going through in Blender moving and scaling UV's to fit.

This process felt awfully repetitive, as well as awfully automatable, cue this script.

It takes in an `.obj` model file with accompanying `.mtl` and analyses what texture files it uses, as well as how much of each texture file is used. With this information, the script optionally crops textures down to just what is used before packing them into a single texture. Finally, a copy of the original `.obj` is output with updated UV coordinates (corresponding to the new, single texture). An updated `.mtl` is also output.

## Dependencies
Written in [*Python 3*](https://www.python.org/downloads/), and using the great [*Pillow*](https://python-pillow.github.io/) image processing/manipulation library.
You can install the former and use the `pip` tool it should bring to grab *Pillow*:

    pip install pillow

And that's all that the script should depend on.

## Usage
Ensure you've got *Python 3* and *Pillow* installed, then nab the packer script (including its subfolder with `imagepacker.py`) by [downloading a ZIP of the repo](https://github.com/theFroh/imagepacker/archive/master.zip).
Extract this archive somewhere you wont forget!

### Packing a model
Drag and drop an `.obj` file onto `objuvpacker.py` and behold the results inside of the `_packed` folder next to your .obj.

#### In more depth:

1. Ensure the model you wish to pack is in the Wavefront OBJ format (I've only tested using Blender exports) and that it has an MTL material definition file
2. Ensure the textures the `.mtl` refers to exist and are accessible (note; the script will check locally for the texture files *first*, before checking the full path), and that they are in a suitable format supported by Pillow (such as: `.tga`, `.jpg`, `.png`, `.bmp`, etc.)
3. Now either;
    - Drag and drop your `.obj` file onto `objuvpacker.py` to use the default settings

    or,

    - Run `python objuvpacker.py [path to your .obj file]` in a terminal (possibly to use the arguments described later)
4. Inspect the packed output `.obj` and texture inside of the output directory (a folder named after the original `.obj` with `_packed` appended to it) to see if everything went well.
5. If you're doing this for Tabletop Simulator, convert the packed texture into a more compressed form (`.jpg`) if you do not need the transparency *(you usually don't)*, and then upload the `.obj` and texture file for use ingame -- have fun!

#### Arguments

- *Material* `-m --material MATERIAL` - Explicitly tell the script what `.mtl` file to use.
- *Output* `-o --output OUTPUT` - Explicitly tell the script where output to.
- *Add* `-a --add [ADD, ...]` - Additional images to be packed. (Probably useless)
- *No crop* `--no-crop` - Disable any cropping or tiling/unrolling.
- *No tile* `--no-tile` - Ignore any wrapped/tiling of textures (depends on cropping).

#### Troubleshooting
If you're having trouble packing a model, you can try running the script in a terminal with `--no-crop` and `--no-wrap`. This will use the simplest possible packing, but should be fairly solid.

#### Tiling or wrapping warnings
You may see a prompt similar to:

    WARNING: The following texture has coordinates that imply it tiles 0.8x10.7 times:
        E:\Syncthing\code\obj texture packer\sample\raider\predator_track_l.tga
    This may be intentional (i.e. tank track textures), or a sign of problematic UV coordinates.
    Consider only unwrapping/tiling this if you know that it is intentional.
    (If you are unsure, just hit enter to answer 'No')
    Do you want to unroll this wrapping by tiling the texture? [y/N]:

Then you should read the prompt. This arises when an object has UV coordinates, or entire islands, outside of the usual range of `[0,1]`, which could be for a few reasons:

1. The texture in question is intended to tile or repeat multiple times, such as track links for a tank. In this case, I'd advise answering `yes` to tiling to avoid having missing textures on the model's tracks.
2. The islands/UV coordinates in question were put outside of the usual space because they aren't used. Answer `no` (or press enter, it defaults to `no`). These shouldn't have a visible effect on the final model.
3. It is a bug, or the modeller has purposefully made use of the UV wrapping behaviour for layout purposes. Answer `no`, check if it causes unwanted texture issues (and isn't a case of `2.`), and if it does produce issues, manually shift the troublesome UV's into place in the *original model (before packing, to save you doing it again later)* using your favourite 3D editor.

In the example message given above, it should be clear that the texture in question is likely a tank track segment, and that this track repeats nearly 11 times vertically -- it would definitely be wise to answer `yes` to unroll the texture so that the model's tracks are fully textured.

##### Why does this happen?
Game and 3D rendering engines tend to treat a texture as infinitely tiling in every direction for the sake of UV coordinates. A UV face that totally encompasses what would usually be the entire texture is effectively filled with repetitions of the texture. UV Faces outside of where the texture usually ends are filled with whatever part of the texture would lie under them if it was repeated to reach.

For example, a very small texture consisting of a single tank track segment. The tank track model itself may be very long, so its UV's may be arranged to continue well beyond the tiny segment's texture space -- instead of not being textured, the engine rendering the tank model simply internally repeats that tiny track segment along the entire track model's UV faces, texturing the entire track with only a tiny actual texture.

I'll have to add a diagram to help illustrate the above, but because the script combines multiple textures into one larger texture, each texture no longer infinitely tiles in each direction, so models which intentionally "wrap" a texture many times must be "unrolled" -- tiled so that the UV faces in question actually have the correct texture to display.

To make things tricky, modelers sometimes intentionally put UV islands outside of the normal space, and sometimes importer bugs or differing coordinate systems can do the same. All of these will look fine in a 3D editor, but if packed without unrolling/wrapping/tiling can sometimes look incorrect.

## Technical details

I'm surprised Blender doesn't have a built in tool that does this.

Nothing special going on here, a simple rectangle packing algorithm is used to pack representations of the cropped textures fairly tightly. The algorithm is not optimal, and is partly vertically biased. There can be a lot of whitespace in some of its solutions, especially as it does not try to rotate rectangles, but this isn't really an issue as whitespace compresses well.

I wasn't planning on releasing this script as is, but I think its usefulness to Warhammer 40k players seeking to bring armies into Tabletop Simulator outweighs waiting until I get around to wrapping this in a nice GUI. The code is fairly haphazard in places, apologies!
