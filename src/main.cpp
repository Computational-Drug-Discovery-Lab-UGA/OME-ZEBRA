#include <ome/files/VariantPixelBuffer.h>
#include <ome/files/in/TIFFReader.h>
#include <ome/files/in/OMETIFFReader.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

using namespace std;

void readPixelData(const FormatReader& reader, std::ostream& stream, std::string filename) {
    // Get total number of images (series)
    dimension_size_type ic = reader.getSeriesCount();
    stream << "Image count: " << ic << '\n';
    vector <int[2048][1024]> matrices;

    // Loop over images
    for (dimension_size_type i = 0 ; i < ic; ++i) {
        reader.setSeries(i);

        // Get total number of planes (for this image index)
        dimension_size_type pc = reader.getImageCount();
        stream << "\tPlane count: " << pc << '\n';

        // Pixel buffer
        VariantPixelBuffer buf;

        // Loop over planes (for this image index)
        for (dimension_size_type p = 0 ; p < pc; ++p) {
          // Read the entire plane into the pixel buffer.
          reader.openBytes(p, buf);

          auto pixelArray = buf.array<uint16_t>();
          int pixelMatrix[2048][1024];

          int i = 0;
  				for (int y = 0; y < 2048; ++y) {
  					for (int x = 0; x < 1024; ++x) {
  						pixelMatrix[y][x] = pixelArray.data()[i];
  						++i;
  					}
  				}

          for(int k=0;k<2048;i++){
            for(int j=0;j<1024;i++) {
              cout << pixelMatrix[k][j] << " ";
            }
            cout << endl;
          }

          matricies.push_back(pixelMatrix);

          stream << "Pixel data for Image " << i
                 << " Plane " << p << " contains "
                 << buf.numElements() << " pixels\n";
        }
    }
}

int main(int argc, char *argv[]) {
    shared_ptr<FormatReader> reader(make_shared<OMETIFFReader>());
    reader->setMetadataFiltered(false);
    reader->setGroupFiles(true);

    if(argc!=2) {
      cout << "Usage: ./exe <file>"
      return 1;
    } else {
      reader->setId(argv[1]);
      readPixelData(*reader, cout, fileName.string());
      reader->close();
    }
    return 0;
}
