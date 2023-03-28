import shapely
from sentinelsat import SentinelAPI


class Get_Data:

    def __init__(self):
        self.username = 'username'
        self.password = 'password'
        self.api = SentinelAPI(self.username, self.password,
                               'https://apihub.copernicus.eu/apihub')

    # Funtion that takes Dictionary with Long/Lat and returns the shapely point
    def makePoints(self, point):
        p = (point["Longitude_Degrees"], point["Latitude_Degrees"])
        return shapely.geometry.Point(p[0], p[1])

    def download_data(self, point):
        # self.footprint = shapely.geometry.Point(72.1378, -6.2779)
        self.footprint = point
        self.products = self.api.query(self.footprint,
                                       date=('20141219', '20221229'), platformname='Sentinel-2',  # noqa
                                       cloudcoverpercentage=(0, 30))
        self.products_df = self.api.to_dataframe(
            self.products)  # Put into Dataframe

        print(f"There are {self.products_df.shape[0]} products currently")
        print()

        for i in range(self.products_df.shape[0]):
            id = self.products_df.uuid[i]  # Get the ID of the first example
            product_info = self.api.get_product_odata(
                id)  # Get the Product Information
            is_online = product_info['Online']  # Check if its online
            if is_online:
                print()
                print(f'Product {id} is online. Starting download.')
                self.api.download(self.products_df.index[i])
                print("Download Complete")
                break  # Break   --> Download the first available data
            else:
                print()
                print(f'Product {id} is not online.')
                self.api.trigger_offline_retrieval(self.products_df.index[i])


if __name__ == '__main__':
    setup = Get_Data()  # Setup the Download
    lat, long = input("What is your latitude and longitude ?  ").split(",")  # Split for faster input  # noqa
    print(f"Your point is: {long} long, {lat} lat.")

    # Setup the point:
    point = {"Longitude_Degrees": long, "Latitude_Degrees": lat}
    result_point = setup.makePoints(point)
    setup.download_data(result_point)
