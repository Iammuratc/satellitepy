Documentation for the FR24 dataset masks

Most airplanes are taken from https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO=LJ60

Different subtypes (like A320_211 and A320_212) a describing:
(Source: https://forum.airliners.de/topic/30723-worin-unterscheiden-sich-untervarianten-der-flugzeugtypen/)
    Different enginetypes for Airbus
    Different airlines for Boeing
Different subtypes have the same masks


### Coordinate transformation
Required parameters:
- x_min, x_max, y_min, y_max for image corners
- Width and height of the image
- The geocoordinates must have the same CRS as the image otherwise this will cause problems!

The pixel coordinates can be calculated with this formular:
```python
x_pixel = round(width * (old_x - x_min) / (x_max - x_min))
y_pixel = round(height * (old_y - y_min) / (y_max - y_min))
```

## How to use the transformation script
You need 2 files:
- The generated GeoJSON file (Layer CRS needs to be exactly like the image CRS!)
- The image

You also need either the IMD file which is sometimes in the same directory as the image OR the QGIS Extent Information for the image. 
You can get the Qgis Extent Information by right clicking on the image -> Properties. Just copy the coordinates as a string and the script will do the rest. 
I would recommend using the QGIS Extent Information because not every satellite image has an IMD file.

The script will show you the image so you can visually check if the points are mapped correctly.
