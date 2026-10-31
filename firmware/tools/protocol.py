"""固件文件接口的常量（assets.py 用）。"""

DEFAULT_IP = "192.168.4.1"     # 固件 SoftAP 的地址（热点名 Xbotics-MiniDuck-XXXX，XXXX = 热点 MAC 后 4 位，开机串口打出全名）
WEB_PORTS = {"remote": 80, "servo": 81, "calib": 82, "files": 83, "debug": 84, "sound": 85}
FS_LIST_PATH = "/fs"            # GET → JSON：files、used、total、writable、why、policies、sounds
FS_UPLOAD_PATH = "/fs/upload"   # POST，请求头 X-Name = 文件名，请求体 = 文件原样；.mdp → /policy/，.wav → /audio/
FS_UPLOAD_NAME_HEADER = "X-Name"  # 文件名走请求头：板上 WebServer 收完请求体才解析网址参数
FS_DELETE_PATH = "/fs/delete"   # POST ?path=/policy/sitstand.mdp（只删上传的文件与上电记录，出厂文件删不掉）
FS_RESTORE_PATH = "/fs/restore" # POST：删掉全部上传的策略与叫声，回复后重启（恢复出厂）
FS_REBOOT_PATH = "/reboot"      # POST，回复后重启（新文件开机才载入）
RECORD_PATH = "/record"         # GET ?session=current|<序号>（调试页 84 口）：上电记录，格式见 boot_record.py
POLICY_HEADER_FMT = "<4sHHII24s16s16siiiifffiI20s"
