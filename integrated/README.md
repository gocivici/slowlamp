
Remember to change file paths in `correct_color_HD108.py` (data folder) and the tuning file in `fgc_integrated.py`

Remember to change user in `slowlamp_app.service`

```
sudo nano /etc/systemd/system/slowlamp_app.service
```
then copy the modified slowlamp_app.service in this folder to the file above

then
```
sudo systemctl enable slowlamp_app.service
sudo systemctl start slowlamp_app.service
```
to stop:
```
sudo systemctl stop slowlamp_app.service
```