from mathx.robotics.database import Database

def test_database():
  max_size=20
  db=Database({"max_size":max_size,"load_factor":.7})

  for i in range(100):
    db.add(i)
    assert db.size()<=max_size
    assert i<10 or db.size()>max_size/2
  
    assert db.data[-1]==i
    assert db.size()<2 or db.data[-2]==i-1

    print(f"i={i} size={db.size()} db[-1]={db.data[-1]} db[-2]={db.data[-2] if db.size()>=2 else None}")
