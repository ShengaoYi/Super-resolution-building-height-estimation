import ee
import geemap

ee.Authenticate()

# Initialize the Earth Engine API
ee.Initialize(project='gee-shengao')
# Define the region of interest (e.g., a bounding box or polygon)
roi = ee.Geometry.Rectangle([-97.8953453913794505,30.1464727882386114, -97.5840791454320424,30.1643403580647382])

# Define the time range
start_date = '2020-01-01'
end_date = '2020-12-31'

# Function to mask clouds for Sentinel-2 images
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)  # Clouds
    cirrus_mask = qa.bitwiseAnd(1 << 11).eq(0)  # Cirrus clouds
    return image.updateMask(cloud_mask).updateMask(cirrus_mask)

# Download Sentinel-1 images
def download_sentinel1(roi, start_date, end_date, output_path=r'./data/S1/S1_Austin_0_0.tif'):
    sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW'))

    # 选择 VV 和 VH 波段
    sentinel1_image = sentinel1.select(['VV', 'VH']).median().clip(roi)

    geemap.ee_export_image(sentinel1_image, filename=output_path, scale=10, region=roi)

    # 指定导出参数, 适用于大文件, 本地导出有大小限制
    # task = ee.batch.Export.image.toDrive(
    #     image=sentinel1_image,
    #     description='Sentinel-1',
    #     folder='GEE_Downloads',
    #     scale=10,
    #     region=roi.getInfo()['coordinates'],
    #     fileFormat='GeoTIFF'
    # )
    # task.start()


# Download Sentinel-2 images
def download_sentinel2(roi, start_date, end_date, output_path=r'./data/S2/S2_Austin_0_0.tif'):
    # 定义需要的 Sentinel-2 波段
    bands = ['B2',  # Blue (10m)
             'B3',  # Green (10m)
             'B4',  # Red (10m)
             'B8',  # NIR (10m)
             'B11', # SWIR1 (20m)
             'B12'] # SWIR2 (20m)

    sentinel2 = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .map(mask_s2_clouds) \
        .select(bands)

    # 获取中值合成图像并裁剪到区域
    sentinel2_image = sentinel2.median().clip(roi)

    geemap.ee_export_image(sentinel2_image, filename=output_path, scale=10, region=roi)

    # 指定导出参数, 适用于大文件, 本地导出有大小限制
    # task = ee.batch.Export.image.toDrive(
    #     image=sentinel2_image,
    #     description='Sentinel-2',
    #     folder='GEE_Downloads',
    #     scale=10,
    #     region=roi.getInfo()['coordinates'],
    #     fileFormat='GeoTIFF'
    # )
    # task.start()

if __name__ == "__main__":
    download_sentinel1(roi, start_date, end_date, output_path=r'./data/s1/s1_Austin_test.tif')
    download_sentinel2(roi, start_date, end_date, output_path=r'./data/s2/s2_Austin_test.tif')
