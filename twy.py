db_info = {dbname: st.session_state["db_name"], 
        user=st.session_state["db_user"],
        password=st.session_state["db_password"],
        host=st.session_state["db_host"]}

print(**db_info)