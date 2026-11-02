#define MyAppName "MWdriver"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Imperial College London"

[Setup]
AppId={{CB877477-E610-4B44-92D4-EA640AFFC229}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\MWdriver
DefaultGroupName=MWdriver
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64
OutputDir=..\release
OutputBaseFilename=MWdriver-Setup-{#MyAppVersion}
UninstallDisplayIcon={app}\MWdriver.exe
SetupLogging=yes
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut for all users"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\MWdriver\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{commonprograms}\MWdriver\MWdriver"; Filename: "{app}\MWdriver.exe"; WorkingDir: "{app}"
Name: "{commondesktop}\MWdriver"; Filename: "{app}\MWdriver.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\MWdriver.exe"; Description: "Launch MWdriver"; Flags: postinstall nowait skipifsilent