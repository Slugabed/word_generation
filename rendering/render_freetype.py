import freetype
from diploma.rendering import RESOURCE_PATH

face = freetype.Face('/home/master/PycharmProjects/tomophantom/diploma/resources/opensans/OpenSans-Italic.ttf')
face.set_char_size(48 * 64)
face.load_char('S')
bitmap = face.glyph.bitmap
print(type(bitmap))
print(bitmap.buffer)
