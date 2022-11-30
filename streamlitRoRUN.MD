#to run 
streamlit run main.py –server.port 80 --server.fileWatcherType none


5. Use Screen
Streamlit will close the session as soon as you exit SSH session. To make it persistence, we need to use screen tool. Once you create session in screen, you can close the terminal and connect the server using new terminal. It will create persistence session.

# screen -S streamlit

Deploy Streamlit app:

# streamlit run main.py –server.port 80

Now close the terminal and connect with server in new terminal and run following command list the screen session.

# screen -list

We have successfully completed with the installation process and also we have deployed our first app.

In this article, we have seen how to install Streamlit on Ubuntu 20.04