import java.util.*;

public class MultiArray05_T {

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
		
		System.out.print("인원수: ");
		int cnt = scan.nextInt();
		
		String[] name = new String[cnt];
		
		int subjectCnt;
		
		String[][] subject = new String[cnt][];
		int[][] jumsu = new int[cnt][];
		double[] avg = new double[cnt];
		
		for(int i=0; i<cnt; i++) { // 인원수
			System.out.print("이름 입력: ");
			name[i] = scan.next();
			
			System.out.print("과목수 입력: ");		
			subjectCnt = scan.nextInt();

			// 다차원 배열에서 가변배열 생성 시, new String[]을 통한 Memory Load 필요
			subject[i] = new String[subjectCnt];
			for(int j=0; j<subjectCnt; j++) {
				System.out.print("과목명 입력: ");
				subject[i][j] = scan.next();					
			}
				
			jumsu[i] = new int[subjectCnt+1]; // total 포함을 위해
			for(int j=0; j<subjectCnt; j++) {
				System.out.print(subject[i][j] + " 점수 입력: ");
				jumsu[i][j] = scan.nextInt();
				
				// Total: jumsu[][]에서 정의
				jumsu[i][subjectCnt] += jumsu[i][j];
			}

			// Avg
			avg[i] = (double)jumsu[i][subjectCnt] / subjectCnt;

			
		} // for_scanner
		
		for(int i=0; i<cnt; i++) {
			System.out.print("이름\t");

			// headline
			for(int j=0; j<subject[i].length; j++) {
			System.out.print(subject[i][j] + "\t");
			}
			
			System.out.println("총점\t평균");
			
			// Data
			System.out.print("\n" + name[i] + "\t");
			for(int j=0; j<jumsu[i].length; j++) {
				System.out.print(jumsu[i][j] + "\t");
			} // for_j
			System.out.print(String.format("%.2f", avg[i]) + "\n");
			
		} // for_i
		
		scan.close();
	}
	
}
