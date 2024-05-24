# Onboarding Checklist

This tutorial page is intended to serve as a (technical) checklist for onboarding new students/employees to the TAMI
project. Note that this *only* applies to the TAMI project. There may be additional task for onboarding based on the
corresponding group/department policy. The general onboarding process is organized via JIRA, please consult with your
onboarding buddy in case of questions.

* Join the TAMI team via [this
  link](https://teams.microsoft.com/l/team/19%3a594cd5c5ae0a4e3d8ed0172330250101%40thread.skype/conversations?groupId=834073c4-0351-4fc1-8d8c-cba18c760653&tenantId=0ae51e19-07c8-4e4b-bb6d-648ee58410f4)
* Join the CR/AIR4.1 teams via [this link](https://teams.microsoft.com/l/team/19%3aa4ddef11b7674998ad70bfdea66c0e06%40thread.skype/conversations?groupId=3ef5f5f5-4cf8-4c96-95e9-d3d4b6d0423b&tenantId=0ae51e19-07c8-4e4b-bb6d-648ee58410f4)

* Invitations to regular meetings, access to infrastructure such as shared folders etc., is mainly organized via distribution lists. Requests to join one of these groups can be organized via the [IT Service](https://service-management.bosch.tech/sp). Search for `Security Groups / Distribution Lists`. We have the following lists

  | Distribution List    | Who | Used for |
  | --- | --- | --- |
  | RNG_CR_TAMI-CORE  | The core TAMI team | Meetings (Standup, planning, review, social events, technical exchange), access to the [TAMI activity sharepoint](https://bosch.sharepoint.com/sites/msteams_7867599-ROBSharepoint/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=UlZQmk&cid=ff0b2e20%2D23b8%2D450f%2D9b3a%2De3a48f12e41d&RootFolder=%2Fsites%2Fmsteams%5F7867599%2DROBSharepoint%2FShared%20Documents%2FROB%20Activity%20Shares%2FROB%2D021%20TAMI&FolderCTID=0x0120007D7C499250A8B84CB3F98AF0E72D81E7)|
  | Rng_CR_ROB-021-phd | All PhD students of the team | Meetings (Standup, planning, review, social events, technical exchange), access to the [TAMI activity sharepoint](https://bosch.sharepoint.com/sites/msteams_7867599-ROBSharepoint/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=UlZQmk&cid=ff0b2e20%2D23b8%2D450f%2D9b3a%2De3a48f12e41d&RootFolder=%2Fsites%2Fmsteams%5F7867599%2DROBSharepoint%2FShared%20Documents%2FROB%20Activity%20Shares%2FROB%2D021%20TAMI&FolderCTID=0x0120007D7C499250A8B84CB3F98AF0E72D81E7) |
  | RNG_CR_TAMI-STUDENTS | The TAMI students (Interns, MSc) | Meetings (social events, technical exchange), access to the [Students folder in the TAMI activity sharepoint](https://bosch.sharepoint.com/sites/msteams_7867599-ROBSharepoint/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=UlZQmk&cid=ff0b2e20%2D23b8%2D450f%2D9b3a%2De3a48f12e41d&FolderCTID=0x0120007D7C499250A8B84CB3F98AF0E72D81E7&id=%2Fsites%2Fmsteams%5F7867599%2DROBSharepoint%2FShared%20Documents%2FROB%20Activity%20Shares%2FROB%2D021%20TAMI%2F50%5FStudents&viewid=77da97f0%2D3e25%2D405f%2Dab5a%2D15ed29339e4c)|
  | RNG_CR_TAMI-GLOBAL | TAMI stakeholders, group leads, ...  | Meetings (Quarterly demo days)|

  Please add yourself (or your student) to the list that corresponds to your role.

  These lists are used for ROB-wide invitations/access as well:

  | Distribution List    | Who | Used for |
  | --- | --- | --- |
  | Rng_CR_ROB-021     | RNG_CR_TAMI-CORE + Rng_CR_ROB-021-phd |  ROB managed, used for access to the [ROB wiki](https://inside-docupedia.bosch.com/confluence/display/CRROB/CR+Strategic+Portfolio+Robotics+Home), invitations to meetings and access to the GPU cluster. |
  | Rng_CR_ROB-021-students   | RNG_CR_TAMI-STUDENTS | ROB managed, e.g., for access to the GPU cluster |
  | Rng_CR_ROB-021-visitors | RNG_CR_TAMI-GLOBAL | ROB managed, invitations to drumbeat meetings |

  * Recurrent meetings will not update immediately, please ask your buddy to forward you the invitations for
    * Standup, Planning and review meetings (only full-time employees)

* Request access to our code, documentation and planning documents on [Github Enterprise](https://github.boschdevcloud.com/).
  Our code [is hosted here](https://github.boschdevcloud.com/orgs/bcai-internal/teams/tami).
  To request access follow these steps:
    1. Check whether you already have access to the BDC by going to [Github Enterprise][https://github.boschdevcloud.com/]. If you can already access,
      that page skip the next step
    2. Request access to the Bosch Development Cloud if not done already. Follow [this
      guide](https://docs.boschdevcloud.com/userguide/index.pdf). This only needs to be
      done once. Sometimes, the link may not take you to the corresponding subsection. You should follow `1.2.1. Get
      started as a user of the BDC`.
      *Note: the "drop down" menu in the document Sec. 1.2.1 (in case Bosch Development Cloud is not visible) refers to the search bar under "Application with Autom. Account Creation", type "Bosch Development Cloud" there*
    3. Assign the concrete roles for TAMI code access by following [this
      guide](https://docs.boschdevcloud.com/user-guide.html#_idm_self_service). The names of the roles are
      `BDC_Artifactory_01_perm135_reader`, `BDC_Artifactory_01_perm20_reader`, `BDC_Artifactory_01_perm56_reader`, `BDC_Github_01_org10_member`, and `BDC_Github_01_org177_member`
    4. You should get a mail notification once this is done. Syncing back the changes to GHE may take up to one day. Until
      then you may not be able to access the TAMI team yet.
    5. Once everything is done you should be able log in to [GHE](https://github.boschdevcloud.com) by clicking log in
      with your windows account. You should see the `bcai-internal` org when clicking on `Your Organizations`
    6. Go to the [TAMI team page](https://github.boschdevcloud.com/orgs/bcai-internal/teams/tami/members) and request
      access by using the corresponding button
    7. Configure access to the repositories on your PC by following the [instructions to install Git Credential
      Manager](https://github.com/GitCredentialManager/git-credential-manager#linux). Then run

     ```bash
     git-credential-manager configure
     git config --global credential.credentialStore secretservice
     ```
     In case you don't have graphical output, e.g. when working over ssh, use the following line instead. Find more infos [here](https://github.com/GitCredentialManager/git-credential-manager/blob/main/docs/credstores.md#gpgpass-compatible-files).

     ```
     git config --global credential.credentialStore gpg
     ```

     When cloning a repository for the first time with

     ```
     git clone https://github.boschdevcloud.com/bcai-internal/<repo_name>.git
     ```

     you will be guided through an automatic authentication process.

     More details on the authentication process can be found
     [here](https://github.com/GitCredentialManager/git-credential-manager/blob/main/docs/credstores.md) and
     [here](https://inside-docupedia.bosch.com/confluence/display/BCAIR/GitHub+Enterprise).

* If you are onboarding a student, please add him/her to [our overview list for
  students](https://bosch.sharepoint.com/%3A.x%3A./r/sites/msteams_ab4fb0/Shared%20Documents/General/AMIRA%20Student%20Projects.xlsx?d=wa38d7874bfdf4d3ba1359b78da6c1ed3&csf=1&web=1&e=BIb2eL)

* (Optional): Get access to the robot lab (Rng121/0 F.010). This is only necessary if the employee/student actually needs
  to work with our hardware. To get access, navigate to the AIR4.1 teams channel and create a task in the [Lab Access
  tab](https://teams.microsoft.com/l/entity/com.microsoft.teamspace.tab.planner/_djb2_msteams_prefix_3898866852?context=%7B%22channelId%22%3A%2219%3Aa4ddef11b7674998ad70bfdea66c0e06%40thread.skype%22%7D&groupId=3ef5f5f5-4cf8-4c96-95e9-d3d4b6d0423b&tenantId=0ae51e19-07c8-4e4b-bb6d-648ee58410f4)
  and assign it to Patrick Kesper.

  NOTE: The task *must* be created by the person who is requesting the access not the supervisor/buddy

  NOTE: All access will removed on 31.03 per default every year. To renew your access, there will be an annual safety
  instructions around mid/end of February.
* (Optional): We use calendars to manage bookings for our robot setups (see [here](using_panda_arm.md) for more
  details). Access is controlled via the distribution list  `RNG_CR_PJ-TAMI-CORE`.
* (Optional): Request sudo rights on OSD PC. This can be done via [IT Service
  Portal](https://rb-servicecatalog.apps.intranet.bosch.com/RequestCenter/website/Grunt/application/search.html?q=Enable%20or%20Disable%20Temporary%20Administrative%20Rights).
  It may be reasonable to also request sudo for the supervisor. Note that the request must be made in the name of the
  *owner* of the PC. For most student resources this will be Jörg Wagner (for
  Renningen) or Vien (for Tübingen).

* (Optional): Access to SocialCoding (currently used only for our data repository) Request access to SocialCoding [here](http://rb-cae.de.bosch.com/ServiceRegistration/?SocialCoding). To get access to the
TAMI code base you have to
  1. Request access to the corresponding AD group via [this IT
      Request](https://rb-servicecatalog.apps.intranet.bosch.com/RequestCenter/website/Grunt/application/offer.html?id=4019).
      In case, the link is not valid for you, the name of the request is `Security Groups / Distribution Lists`:
      * Choose `Order for others -> Patrick Kesper`. A new request form should open
      * Use `Request/Remove membership` from the option dialog
      * Select `rb_sde_soco_amira_developerwrite_uf` as security group (the security group might not show up in the dropdown menu, but you can search for it in the search field)
      * Use the `Add member` to request access. Usually, the request should be approved within 24 hours.
  2. After the request has been approved, you can create an ssh key with the command
    ```
    local$ ssh-keygen
    ```
    Now copy the output of
    ```
    cat ~/.ssh/id_rsa.pub
    ```
    and add it to SoCo by clicking on your user symbol top right, then click  `Manage account->SSH keys->Add key`.
    Now you should be able to clone a social coding repository without further authentication from this machine.

## FAQ:
* **Q**: I have requested all roles, but still cannot see any organization on
  GHE<br>
  **A**: You may have to request BIOS access explicitly via the [SocialCoding
  form]([here](http://rb-cae.de.bosch.com/ServiceRegistration/?SocialCoding).
  After that has been approved allow one night for syncing the changes

