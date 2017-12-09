import dateutil.parser
import time
dt = dateutil.parser.parse("6/21/2017")
print(time.mktime(dt.timetuple()))