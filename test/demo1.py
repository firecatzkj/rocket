class Car:
    def __init__(self, year, model, make):
        self.year = year
        self.model = model
        self.make = make
        self.odometer_reading = 0

    def update_odometer(self, mileage):
        self.odometer_reading = mileage

    def descriptive_name(self):
        long_name = str(self.year)+' '+self.make.title()+' '+self.model.title()
        print(long_name)

    def reading_odemeter(self):
        print("The car has "+str(self.odometer_reading)+" on it.")

my_new_car = Car('2016', 'a4', 'audi')
my_new_car.update_odometer(728)
my_new_car.descriptive_name()
my_new_car.reading_odemeter()
