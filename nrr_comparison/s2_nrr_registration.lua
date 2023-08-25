function scandir(directory)
  local i, t, popen = 0, {}, io.popen
  for filename in popen('dir "'..directory..'" /o:n /b'):lines() do
    i = i + 1
    t[i] = string.gsub(filename, '.nii', '')
  end
  return t
end

-- set path here
path = [[ ]]
struct_path = [[ ]]
out_path = [[ ]]
patient_ids = scandir(path)

structs = {"Brainstem", "Mandible", "Parotid-Lt", "Parotid-Rt", "Spinal-Cord", "Submandibular-Lt", "Submandibular-Rt"}

for rdx, ref_pat_id in pairs(patient_ids) do
  wm.Scan[1]:read_nifty(path .. ref_pat_id .. '.nii')
  print("Registering to patient: " .. ref_pat_id .. " - num: " .. rdx .. "/34")
  for _, patient_id in pairs(patient_ids) do
    wm.Scan[2]:read_nifty(path .. patient_id .. '.nii')

    -- perform NRR
    wm.scan[2]:niftyreg_f3d(wm.scan[1],nil,"-ln 5 -lp 4 -be 0.001 -smooR 1 -smooF 1 -jl 0.0001")

    wm.Scan[4]=wm.Scan[2]:as(wm.Scan[1]) -- def ref image

    for sdx, structure_name in pairs(structs) do
      wm.Scan[3]:read_nifty(struct_path .. patient_id .. '_' .. structure_name .. '.nii')
      wm.Scan[2].data = wm.Scan[3].data
      wm.Scan[5] = wm.Scan[2]:as(wm.Scan[1]) -- def struct
      wm.Scan[5]:write_nifty(out_path .. '/' .. ref_pat_id .. '/' .. patient_id .. '_' .. structure_name .. '.nii')
    end
  end
end