import datetime

created_at_timestamp = 1710048269
finished_at_timestamp = 1710051914

# 将时间戳转换为 datetime 对象
created_at_datetime = datetime.datetime.utcfromtimestamp(created_at_timestamp)
finished_at_datetime = datetime.datetime.utcfromtimestamp(finished_at_timestamp)

# 将 datetime 对象格式化为易读的字符串
created_at_str = created_at_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')
finished_at_str = finished_at_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')

print(f"Created at: {created_at_str}")
print(f"Finished at: {finished_at_str}")
