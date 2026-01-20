
Remember to change file paths in `config.json` data folder and tuning file (= "" if there isn't any tuning file)

Remember to change user and paths in `slowlamp_app.service`

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