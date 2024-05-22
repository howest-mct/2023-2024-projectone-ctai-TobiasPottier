# Project One: Deploying

## Instructions for Deploying (RPi)

### ⚠️ DO IT YOURSELF: Automatically Start Your Project When It's Finished

1. **Create a File Named _myproject.service_:**

   - Create a new file and name it `myproject.service`.

2. **Place the Following Code in the File:**

   ```ini
   [Unit]
   Description=ProjectOne Project
   After=network.target

   [Service]
   ExecStart=/home/user/<name_of_your_repo>/<venv>/bin/python -u /home/user/<name_of_your_repo>/RPi/app.py
   WorkingDirectory=/home/user/<name_of_your_repo>/RPi
   StandardOutput=inherit
   StandardError=inherit
   Restart=always
   User=user
   CPUSchedulingPolicy=rr
   CPUSchedulingPriority=99

   [Install]
   WantedBy=multi-user.target
   ```

3. **Copy This File to _/etc/systemd/system_ as Root User:**

   - Use the command:
     ```sh
     sudo cp myproject.service /etc/systemd/system/myproject.service
     ```

4. **Test the File by Starting It:**

   - Start the service with:
     ```sh
     sudo systemctl start myproject.service
     ```

5. **Stop the Service:**

   - Stop the service with:
     ```sh
     sudo systemctl stop myproject.service
     ```

6. **Enable the Script to Start Automatically After Booting:**

   - Enable the service to start automatically with:
     ```sh
     sudo systemctl enable myproject.service
     ```

7. **Check the Status of Your Service:**

   - View the service status with:
     ```sh
     sudo service mijnproject status
     ```

8. **View the Logs:**
   - Check the logs with:
     ```sh
     sudo journalctl -u mijnproject
     ```
