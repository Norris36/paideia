# Authentication overview

Authentication is done by Azure authentication service for app service.
Each page in the streamlit application must check that the user is authorized by adding the 
`auth.authenticate()` function. 

The function checks if the user is member of a group that is allowed to access the application. 
By default the set of allowed groups is fetched from the file `allowed_groups.txt` in the root of the application.


# Restrict access to certain pages
It is possible to call `auth.authenticate()` with a set of allowed groups. E.g. 
    auth.authenticate({'8caf8a22-e340-4f14-b59b-edfb0a054913', 'd35b59a9-485c-4c71-b82b-0651cec8cb68'})

This will bypass the normal authorization (allowed_groups.txt) and only check if the user is member of one of the groups passed to the function. 

This can be used to restrict access to certain pages. It is also possible to pass a message to the function that will be shown to users without access. 


# Local development
To allow local development, set the environment variable `ALLOW_ANONYMOUS_USERS` to `TRUE`
This will bypass the authentication. 



# Setup Azure authentication for app-service

The streamlit application must have authentication enabled in Azure.

Authentication is added from the Azure portal and Microsoft is selected as Identity provider. 
The application is registered in the AAD
 - Name:`Dcei-GPT-Explorer`
 - AppID: `33e474f7-e369-4297-93f4-462b0550e64d`

Redirect URIs must match the URL of the deployment. E.g.:
 - https://gnpt-explorer.azurewebsites.net/.auth/login/aad/callback
 - https://gnpt-explorer-development.azurewebsites.net/.auth/login/aad/callback

